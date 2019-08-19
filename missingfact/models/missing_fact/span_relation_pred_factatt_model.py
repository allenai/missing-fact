import logging
from typing import Dict, Optional, List, Any

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, MatrixAttention, InputVariationalDropout
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import replace_masked_values, masked_softmax, combine_tensors, \
    weighted_sum, masked_mean
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import softmax

from missingfact.models.missing_fact.utils import add_relation_predictions, add_tuple_predictions
from missingfact.models.missing_fact.utils import get_agg_rep
from missingfact.models.missing_fact.utils import get_embedding
from missingfact.models.missing_fact.utils import get_text_representation

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.WARN)


@Model.register("span_relation_pred_factatt")
class SpanRelationPredFactAttModel(Model):
    """
    This ``Model`` implements the main answer span + relation based model (KGG) described in
    What's Missing: A Knowledge Gap Guided Approach for Multi-hop Question Answering (EMNLP '19)
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 coverage_ff: FeedForward,
                 relation_predictor: FeedForward,
                 scale_relation_loss: float = 1.0,
                 aggregate: str = "max",
                 combination: str = "x,y",
                 answer_choice_combination: Optional[str] = None,
                 coverage_combination: Optional[str] = None,
                 var_dropout: float = 0.0,
                 use_projection: bool = False,
                 ignore_spans: bool = True,
                 ignore_relns: bool = False,
                 ignore_ann: bool = False,
                 span_extractor: Optional[SpanExtractor] = None,
                 reln_ff: Optional[FeedForward] = None,
                 attention: Optional[MatrixAttention] = None,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        """
        :param vocab: AllenNLP Vocabulary
        :param text_field_embedder: AllenNLP Textfield embedder
        :param coverage_ff: Feedforward network that computes the "Fact-Relevance" score_f i.e. how
        well does the fact "cover" the question + answer
        :param relation_predictor: Feedforward network that predicts the relation label R_j
        :param scale_relation_loss: Scalar used to scale the relation loss term, \lambda
        :param aggregate: Pooling function used to aggregate question/fact vector representations in
         "Relation Prediction Score". Choices: max, avg, last
        :param combination: Combination string used to combine vector representation \bigotimes
        :param answer_choice_combination: If set, use this combination string instead of combination
        for combining the answer-based and choice-based fact representation
        :param coverage_combination: If set, use this combination string instead of combination
        for combining the question-choice-based fact rep and fact rep
        :param var_dropout: Variational dropout probability on the input embeddings
        :param use_projection: If set to true, learn a projector to map relation representations to
        a #rel-dimensional vector. Otherwise, the relation predictor should produce embeddings that
        match the #rels.
        :param ignore_spans: If set to true, don't use span representation of the answers in the
        fact_choice_question_rep (default: true)
        :param ignore_relns: If set to true, don't use the relation labels/scores (no relation
        representations computed or scored)
        :param ignore_ann: If set to true, ignore all auxilliary annotation i.e. spans and relations
        Use the entire fact to compute answer span-based representations. No loss computed against
        the relation label. Note that latent relation representations will still be computed
        :param span_extractor: SpanExtractor used to compute answer span representation
        :param reln_ff: Feedforward used to calculate the relation prediction score
        :param attention: Attention function used
        :param encoder: Encoder used to convert seq of word embeddings into contextual (e.g. LSTM)
        representations
        :param initializer: Initializer used for parameters
        """
        super(SpanRelationPredFactAttModel, self).__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._coverage_ff = coverage_ff
        if attention:
            self._attention = attention
        else:
            self._attention = DotProductMatrixAttention()
        if var_dropout > 0.0:
            self._var_dropout = InputVariationalDropout(var_dropout)
        else:
            self._var_dropout = None

        self._num_relations = vocab.get_vocab_size(namespace="relation_labels")

        self._ignore_spans = ignore_spans
        self._aggregate = aggregate
        self._scale_relation_loss = scale_relation_loss
        if span_extractor is None and not ignore_spans:
            raise ConfigurationError("ignore_spans set to False but no span_extractor provided!")
        self._span_extractor = span_extractor
        self._relation_predictor = relation_predictor
        # simple projector
        if use_projection:
            self._relation_projector = torch.nn.Linear(self._relation_predictor.get_output_dim(),
                                                       self._num_relations)
        else:
            self._relation_projector = None
        self._combination = combination
        if answer_choice_combination:
            self._answer_choice_combination = answer_choice_combination
        else:
            self._answer_choice_combination = combination

        if coverage_combination:
            self._coverage_combination = coverage_combination
        else:
            self._coverage_combination = combination
        self._ignore_ann = ignore_ann
        self._ignore_relns = ignore_relns
        if reln_ff is None and not ignore_relns:
            raise ConfigurationError("ignore_relns set to False but no reln_ff provided!")
        self._reln_ff = reln_ff
        self._encoder = encoder
        self._aggr_label_accuracy = BooleanAccuracy()
        self._aggr_choice_accuracy = CategoricalAccuracy()
        self._relation_loss = torch.nn.BCEWithLogitsLoss()
        self._choice_loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def get_text_representation(self, textfield, wrapping_dims):
        return get_text_representation(textfield, wrapping_dims, self._text_field_embedder,
                                       self._encoder, self._aggregate)

    def merge_dimensions(self, input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError("No dimension to distribute: " + str(input_size))

        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, input_size).
        squashed_shape = [-1] + [x for x in input_size[2:]]
        return input_tensor.contiguous().view(*squashed_shape)

    def add_dimension(self, input_tensor, dim, num):
        """
        Expands the input tensor by introducing an additional dimension at dim with size num
        """
        input_size = input_tensor.size()
        if dim < 0:
            dim = len(input_size) + dim + 1
        output_size = [x for x in input_size[0:dim]] + [num] + [x for x in input_size[dim:]]
        return input_tensor.unsqueeze(dim).expand(output_size)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                choices_list: Dict[str, torch.LongTensor],
                choice_kb: Dict[str, torch.LongTensor],
                answer_text: Dict[str, torch.LongTensor],
                fact: Dict[str, torch.LongTensor],
                answer_spans: torch.IntTensor,
                relations: torch.IntTensor = None,
                relation_label: torch.IntTensor = None,
                answer_id: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # B X C X Ct X D
        embedded_choice, choice_mask = get_embedding(choices_list, 1, self._text_field_embedder,
                                                     self._encoder, self._var_dropout)
        # B X C X D
        # agg_choice, agg_choice_mask = get_agg_rep(embedded_choice, choice_mask, 1, self._encoder, self._aggregate)
        num_choices = embedded_choice.size()[1]
        batch_size = embedded_choice.size()[0]
        # B X Qt X D
        embedded_question, question_mask = get_embedding(question, 0, self._text_field_embedder,
                                                         self._encoder, self._var_dropout)
        # B X D
        agg_question, agg_question_mask = get_agg_rep(embedded_question, question_mask, 0,
                                                      self._encoder, self._aggregate)

        # B X Ft X D
        embedded_fact, fact_mask = get_embedding(fact, 0, self._text_field_embedder, self._encoder,
                                                 self._var_dropout)
        # B X D
        agg_fact, agg_fact_mask = get_agg_rep(embedded_fact, fact_mask, 0, self._encoder,
                                              self._aggregate)

        # ==============================================
        # Interaction between fact and question
        # ==============================================
        # B x Ft x Qt
        fact_question_att = self._attention(embedded_fact, embedded_question)
        fact_question_mask = self.add_dimension(question_mask, 1, fact_question_att.shape[1])
        masked_fact_question_att = replace_masked_values(fact_question_att,
                                                         fact_question_mask, -1e7)
        # B X Ft
        fact_question_att_max = masked_fact_question_att.max(dim=-1)[0].squeeze(-1)
        fact_question_att_softmax = masked_softmax(fact_question_att_max, fact_mask)
        # B X D
        fact_question_att_rep = weighted_sum(embedded_fact, fact_question_att_softmax)
        # B*C X D
        cmerged_fact_question_att_rep = self.merge_dimensions(
            self.add_dimension(fact_question_att_rep, 1, num_choices))

        # ==============================================
        # Interaction between fact and answer choices
        # ==============================================

        # B*C X Ft X D
        cmerged_embedded_fact = self.merge_dimensions(
            self.add_dimension(embedded_fact, 1, num_choices))
        cmerged_fact_mask = self.merge_dimensions(self.add_dimension(fact_mask, 1, num_choices))

        # B*C X Ct X D
        cmerged_embedded_choice = self.merge_dimensions(embedded_choice)
        cmerged_choice_mask = self.merge_dimensions(choice_mask)

        # B*C X Ft X Ct
        cmerged_fact_choice_att = self._attention(cmerged_embedded_fact, cmerged_embedded_choice)
        cmerged_fact_choice_mask = self.add_dimension(cmerged_choice_mask, 1,
                                                      cmerged_fact_choice_att.shape[1])
        masked_cmerged_fact_choice_att = replace_masked_values(cmerged_fact_choice_att,
                                                               cmerged_fact_choice_mask, -1e7)

        # B*C X Ft
        cmerged_fact_choice_att_max = masked_cmerged_fact_choice_att.max(dim=-1)[0].squeeze(-1)
        cmerged_fact_choice_att_softmax = masked_softmax(cmerged_fact_choice_att_max,
                                                         cmerged_fact_mask)

        # B*C X D
        cmerged_fact_choice_att_rep = weighted_sum(cmerged_embedded_fact,
                                                   cmerged_fact_choice_att_softmax)

        # ==============================================
        # Combined fact + choice + question + span rep
        # ==============================================
        if not self._ignore_spans and not self._ignore_ann:
            # B X A
            per_span_mask = (answer_spans >= 0).long()[:, :, 0]
            # B X A X D
            per_span_rep = self._span_extractor(embedded_fact, answer_spans, fact_mask,
                                                per_span_mask)
            # expanded_span_mask = per_span_mask.unsqueeze(-1).expand_as(per_span_rep)

            # B X D
            answer_span_rep = per_span_rep[:, 0, :]

            # B*C X D
            cmerged_span_rep = self.merge_dimensions(
                self.add_dimension(answer_span_rep, 1, num_choices))
            fact_choice_question_rep = (cmerged_fact_choice_att_rep +
                                        cmerged_fact_question_att_rep +
                                        cmerged_span_rep) / 3

        else:
            fact_choice_question_rep = (cmerged_fact_choice_att_rep +
                                        cmerged_fact_question_att_rep) / 2
        # B*C X D
        cmerged_fact_rep = masked_mean(cmerged_embedded_fact,
                                       cmerged_fact_mask.unsqueeze(-1).expand_as(
                                           cmerged_embedded_fact),
                                       1)
        # B*C X D
        fact_question_combined_rep = combine_tensors(self._coverage_combination,
                                                     [fact_choice_question_rep, cmerged_fact_rep])

        # B X C X  D
        new_size = [batch_size, num_choices, -1]
        fact_question_combined_rep = fact_question_combined_rep.contiguous().view(*new_size)
        # B X C
        coverage_score = self._coverage_ff(fact_question_combined_rep).squeeze(-1)
        logger.info("coverage_score" + str(coverage_score.shape))

        # ==============================================
        # Interaction between spans+choices and KB
        # ==============================================

        # B X C X K X Kt x D
        embedded_choice_kb, choice_kb_mask = get_embedding(choice_kb, 2, self._text_field_embedder,
                                                           self._encoder, self._var_dropout)
        num_kb = embedded_choice_kb.size()[2]

        # B X A X At X D
        embedded_answer, answer_mask = get_embedding(answer_text, 1, self._text_field_embedder,
                                                     self._encoder, self._var_dropout)
        # B X At X D
        embedded_answer = embedded_answer[:, 0, :, :]
        answer_mask = answer_mask[:, 0, :]

        # B*C*K X Kt X D
        ckmerged_embedded_choice_kb = self.merge_dimensions(self.merge_dimensions(
            embedded_choice_kb)
        )
        ckmerged_choice_kb_mask = self.merge_dimensions(self.merge_dimensions(choice_kb_mask))

        # B*C X At X D
        cmerged_embedded_answer = self.merge_dimensions(self.add_dimension(embedded_answer,
                                                                           1, num_choices))
        cmerged_answer_mask = self.merge_dimensions(self.add_dimension(answer_mask,
                                                                       1, num_choices))
        # B*C*K X At X D
        ckmerged_embedded_answer = self.merge_dimensions(self.add_dimension(cmerged_embedded_answer,
                                                                            1, num_kb))
        ckmerged_answer_mask = self.merge_dimensions(self.add_dimension(cmerged_answer_mask,
                                                                        1, num_kb))
        # B*C*K X Ct X D
        ckmerged_embedded_choice = self.merge_dimensions(self.add_dimension(cmerged_embedded_choice,
                                                                            1, num_kb))
        ckmerged_choice_mask = self.merge_dimensions(self.add_dimension(cmerged_choice_mask,
                                                                        1, num_kb))
        logger.info("ckmerged_choice_mask" + str(ckmerged_choice_mask.shape))

        # == KB rep based on answer span ==
        if self._ignore_ann:
            # B*C*K X Ft X D
            ckmerged_embedded_fact = self.merge_dimensions(self.add_dimension(
                cmerged_embedded_fact, 1, num_kb))
            ckmerged_fact_mask = self.merge_dimensions(self.add_dimension(
                cmerged_fact_mask, 1, num_kb))
            # B*C*K X Kt x Ft
            ckmerged_kb_fact_att = self._attention(ckmerged_embedded_choice_kb,
                                                   ckmerged_embedded_fact)
            ckmerged_kb_fact_mask = self.add_dimension(ckmerged_fact_mask, 1,
                                                       ckmerged_kb_fact_att.shape[1])
            masked_ckmerged_kb_fact_att = replace_masked_values(ckmerged_kb_fact_att,
                                                                ckmerged_kb_fact_mask, -1e7)

            # B*C*K X Kt
            ckmerged_kb_answer_att_max = masked_ckmerged_kb_fact_att.max(dim=-1)[0].squeeze(-1)
        else:
            # B*C*K X Kt x At
            ckmerged_kb_answer_att = self._attention(ckmerged_embedded_choice_kb,
                                                     ckmerged_embedded_answer)
            ckmerged_kb_answer_mask = self.add_dimension(ckmerged_answer_mask, 1,
                                                         ckmerged_kb_answer_att.shape[1])
            masked_ckmerged_kb_answer_att = replace_masked_values(ckmerged_kb_answer_att,
                                                                  ckmerged_kb_answer_mask, -1e7)

            # B*C*K X Kt
            ckmerged_kb_answer_att_max = masked_ckmerged_kb_answer_att.max(dim=-1)[0].squeeze(-1)

        ckmerged_kb_answer_att_softmax = masked_softmax(ckmerged_kb_answer_att_max,
                                                        ckmerged_choice_kb_mask)

        # B*C*K X D
        kb_answer_att_rep = weighted_sum(ckmerged_embedded_choice_kb,
                                         ckmerged_kb_answer_att_softmax)

        # == KB rep based on answer choice ==
        # B*C*K X Kt x Ct
        ckmerged_kb_choice_att = self._attention(ckmerged_embedded_choice_kb,
                                                 ckmerged_embedded_choice)
        ckmerged_kb_choice_mask = self.add_dimension(ckmerged_choice_mask, 1,
                                                     ckmerged_kb_choice_att.shape[1])
        masked_ckmerged_kb_choice_att = replace_masked_values(ckmerged_kb_choice_att,
                                                              ckmerged_kb_choice_mask, -1e7)

        # B*C*K X Kt
        ckmerged_kb_choice_att_max = masked_ckmerged_kb_choice_att.max(dim=-1)[0].squeeze(-1)
        ckmerged_kb_choice_att_softmax = masked_softmax(ckmerged_kb_choice_att_max,
                                                        ckmerged_choice_kb_mask)

        # B*C*K X D
        kb_choice_att_rep = weighted_sum(ckmerged_embedded_choice_kb,
                                         ckmerged_kb_choice_att_softmax)

        # B*C*K X D
        answer_choice_kb_combined_rep = combine_tensors(self._answer_choice_combination,
                                                        [kb_answer_att_rep, kb_choice_att_rep])
        logger.info("answer_choice_kb_combined_rep" + str(answer_choice_kb_combined_rep.shape))

        # ==============================================
        # Relation Predictions
        # ==============================================

        # B*C*K x R
        choice_kb_relation_rep = self._relation_predictor(answer_choice_kb_combined_rep)
        new_choice_kb_size = [batch_size * num_choices, num_kb, -1]
        # B*C*K
        merged_choice_kb_mask = (torch.sum(ckmerged_choice_kb_mask, dim=-1) > 0).float()
        if self._num_relations and not self._ignore_ann:
            if self._relation_projector:
                choice_kb_relation_pred = self._relation_projector(choice_kb_relation_rep)
            else:
                choice_kb_relation_pred = choice_kb_relation_rep

            # Aggregate the predictions
            # B*C*K
            choice_kb_relation_mask = self.add_dimension(
                merged_choice_kb_mask,
                -1,
                choice_kb_relation_pred.shape[-1])
            choice_kb_relation_pred_masked = replace_masked_values(choice_kb_relation_pred,
                                                                   choice_kb_relation_mask,
                                                                   -1e7)
            # B*C X K X R
            relation_pred_perkb = choice_kb_relation_pred_masked.contiguous().view(
                *new_choice_kb_size)
            # B*C X R
            relation_pred_max = relation_pred_perkb.max(dim=1)[0].squeeze(1)

            # B X C X R
            choice_relation_size = [batch_size, num_choices, -1]
            relation_label_logits = relation_pred_max.contiguous().view(*choice_relation_size)
            relation_label_probs = softmax(relation_label_logits, dim=-1)
            # B X C
            add_relation_predictions(self.vocab, relation_label_probs, metadata)
            # B X C X K X R
            choice_kb_relation_size = [batch_size, num_choices, num_kb, -1]
            relation_predictions = choice_kb_relation_rep.contiguous().view(
                *choice_kb_relation_size)
            add_tuple_predictions(relation_predictions, metadata)
            logger.info("relation_predictions" + str(relation_predictions.shape))
        else:
            relation_label_logits = None
            relation_label_probs = None

        if not self._ignore_relns:
            # B X C X D
            expanded_size = [batch_size, num_choices, -1]
            # Aggregate the relation representation
            if self._relation_projector or self._num_relations == 0 or self._ignore_ann:
                # B*C X K X D
                relation_rep_perkb = choice_kb_relation_rep.contiguous().view(*new_choice_kb_size)
                # B*C*K X D
                merged_relation_rep_mask = self.add_dimension(
                    merged_choice_kb_mask,
                    -1,
                    relation_rep_perkb.shape[-1])
                # B*C X K X D
                relation_rep_perkb_mask = merged_relation_rep_mask.contiguous().view(
                    *relation_rep_perkb.size())
                # B*C X D
                agg_relation_rep = masked_mean(relation_rep_perkb, relation_rep_perkb_mask, dim=1)
                # B X C X D
                expanded_relation_rep = agg_relation_rep.contiguous().view(*expanded_size)
            else:
                expanded_relation_rep = relation_label_logits

            expanded_question_rep = agg_question.unsqueeze(1).expand(expanded_size)
            expanded_fact_rep = agg_fact.unsqueeze(1).expand(expanded_size)
            question_fact_rep = combine_tensors(self._combination,
                                                [expanded_question_rep, expanded_fact_rep])

            relation_score_rep = torch.cat([question_fact_rep, expanded_relation_rep], dim=-1)
            relation_score = self._reln_ff(relation_score_rep).squeeze(-1)
            choice_label_logits = (coverage_score + relation_score) / 2
        else:
            choice_label_logits = coverage_score
        logger.info("choice_label_logits" + str(choice_label_logits.shape))

        choice_label_probs = softmax(choice_label_logits, dim=-1)
        output_dict = {"label_logits": choice_label_logits,
                       "label_probs": choice_label_probs,
                       "metadata": metadata}
        if relation_label_logits is not None:
            output_dict["relation_label_logits"] = relation_label_logits
            output_dict["relation_label_probs"] = relation_label_probs

        if answer_id is not None or relation_label is not None:
            self.compute_loss_and_accuracy(answer_id, relation_label, relation_label_logits,
                                           choice_label_logits, output_dict)
        return output_dict

    def compute_loss_and_accuracy(self, answer_id, relation_label, relation_label_logits,
                                  choice_label_logits, output_dict):
        loss = None
        if relation_label is not None and answer_id is not None and relation_label_logits is not None:
            batch_size = answer_id.size()[0]
            # B X 1 x R
            expanded_answer_indices = answer_id.unsqueeze(-1).unsqueeze(-1).expand(
                [batch_size, 1, self._num_relations])
            # B
            relation_mask = (torch.sum(relation_label, dim=-1) > 0).float()
            # B X C X R
            expanded_relation_mask = relation_mask.unsqueeze(1).unsqueeze(2).expand_as(
                relation_label_logits)

            # B X C X R
            labelled_relation_mask = relation_label.unsqueeze(1).expand_as(relation_label_logits)

            relation_label_perchoice = torch.zeros(labelled_relation_mask.size())
            if torch.cuda.is_available():
                relation_label_perchoice = relation_label_perchoice.cuda()
            # All zeros for incorrect choices and true relation labels for correct choices
            relation_label_perchoice.scatter_(1, expanded_answer_indices, labelled_relation_mask)


            # mask out the label logits for the unmarked relations
            combined_mask = labelled_relation_mask.clone()
            mask_correct_choices = torch.ones(labelled_relation_mask.size())
            if torch.cuda.is_available():
                mask_correct_choices = mask_correct_choices.cuda()
            # True relation labels for incorrect choices and all ones for correct choices
            combined_mask.scatter_(1, expanded_answer_indices, mask_correct_choices)
            # Also zero out questions with no marked relations
            combined_mask = replace_masked_values(combined_mask, expanded_relation_mask, 0)
            # first replace all zero-ed relations with -1e7 which will result in prob=0 in the bce loss
            masked_relation_logits = replace_masked_values(relation_label_logits, combined_mask,
                                                           -1e7)

            loss = self._scale_relation_loss * self._relation_loss(masked_relation_logits,
                                                                   relation_label_perchoice)
            # compress B x C X R to get per relation accuracy
            collapsed_label_predictions = (relation_label_logits > 0).float().view([-1, 1])
            collapse_relation_labels = relation_label_perchoice.view([-1, 1])
            collapsed_mask = combined_mask.view([-1, 1]).byte()
            if torch.sum(collapsed_mask).item() > 0:
                self._aggr_label_accuracy(collapsed_label_predictions[collapsed_mask],
                                          collapse_relation_labels[collapsed_mask])

        if answer_id is not None:
            # B X C
            if loss is None:
                loss = self._choice_loss(choice_label_logits, answer_id)
            else:
                loss += self._choice_loss(choice_label_logits, answer_id)
            self._aggr_choice_accuracy(choice_label_logits, answer_id, (answer_id >= 0).float())

            output_dict["loss"] = loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self._aggr_label_accuracy._total_count == 0:
            self._aggr_label_accuracy._total_count = 1
        return {
            'label_accuracy': self._aggr_label_accuracy.get_metric(reset),
            'choice_accuracy': self._aggr_choice_accuracy.get_metric(reset),
        }

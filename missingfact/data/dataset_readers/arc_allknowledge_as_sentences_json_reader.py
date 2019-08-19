import json
import logging
from collections import Counter
from typing import Dict, List, Any

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.reading_comprehension.util import char_span_to_token_span
from allennlp.data.fields import Field, TextField, LabelField, ListField, MetadataField, \
    MultiLabelField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

from missingfact.data.tools.conceptnet_utils import convert_entity_to_string
from missingfact.data.tools.conceptnet_utils import retrieve_scored_tuples, \
    convert_relation_to_string, \
    load_kbtuples_map
from missingfact.data.tools.conceptnet_utils import tokenize_and_stem_str
from missingfact.data.tools.es_search import EsSearch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("arc-knowledge-sentences-json")
class ArcKnowledgeSentencesJsonReader(DatasetReader):
    """
    Reads a file from the AllenAI-V1-Feb2018 dataset in Json format.  This data is
    formatted as jsonl, one json-formatted instance per line.  An example of the json in the data is:

        {"id":"MCAS_2000_4_6",
        "question":{"stem":"Which technology was developed most recently?",
            "choices":[
                {"text":"cellular telephone","label":"A"},
                {"text":"television","label":"B"},
                {"text":"refrigerator","label":"C"},
                {"text":"airplane","label":"D"}
            ]},
        "fact1": "..."
        "answerKey":"A"
        }
    The reader will retrieve relevant sentences from the ElasticSearch instance (if not provided by
    the "prefetched_sentences" key) and relevant tuples from ConceptNet. These sentences+tuples are
    added as fields to the Instance
    """

    def __init__(self,
                 use_elastic_search: bool,
                 use_conceptnet: bool,
                 es_client: str = "localhost",
                 indices: str = "arc_corpus",
                 max_question_length: int = 1000,
                 max_hits_retrieved: int = 500,
                 max_hit_length: int = 300,
                 max_hits_per_choice: int = 100,
                 conceptnet_kb_path: str = "cached_conceptnet.tsv",
                 ignore_related: bool = False,
                 add_relation_labels: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 fact_key: str = None,
                 ignore_spans: bool = False,
                 aggressive_filtering: bool = False,
                 max_tuples: int = 50,
                 use_top_relation: bool = False) -> None:
        """

        :param use_elastic_search: Set to true, if elasticsearch should be used to retrieve relevant
        sentences
        :param use_conceptnet: Set to true, if conceptnet tuples should be used
        :param es_client: ElasticSearch host
        :param indices: ElasticSearch indices (comma-separated)
        :param max_question_length: Question trimmed to these many characters in the ES query
        :param max_hits_retrieved: Maximum number of hits requested per query
        :param max_hit_length: Maximum length of sentence retrieved using ES
        :param max_hits_per_choice: Maximum number of sentences retrieved per choice
        :param conceptnet_kb_path: Location of the ConceptNet tuple TSV
        Format: /r/ConceptNetRel    /c/en/Subject   /c/en/Object
        :param ignore_related: Ignore /r/RelatedTo relations from ConceptNet
        :param add_relation_labels: If set to False, ignore the missing-fact relations (provided
        using the "relations" key)
        :param tokenizer: AllenNLP tokenizer parameter
        :param token_indexers: AllenNLP token indexer parameters
        :param fact_key: The key in the JSONL used to indicate the partial context
        :param ignore_spans: Ignore the answer spans (provided using the "answer_spans" key) in the
        retrieval step. Note the spans can still be used in the model
        :param aggressive_filtering: If set to true, filter near-duplicate sentences too (as
        specified in filter_near_duplicates()
        :param max_tuples: Maximum number of conceptnet tuples
        :param use_top_relation: If set to True, only use the most common relations in the
        annotation as relation labels. If set to false, all of the relations are used as labels.
        """
        super().__init__()
        self._use_elastic_search = use_elastic_search
        self._use_conceptnet = use_conceptnet
        if self._use_elastic_search:
            self._es_search = EsSearch(es_client, indices, max_question_length, max_hits_retrieved,
                                       max_hit_length, max_hits_per_choice)
            self._indices = indices
        if self._use_conceptnet:
            self._kb_tuples, self._tok_idx_map = load_kbtuples_map(conceptnet_kb_path,
                                                                   ignore_related)
        self._max_hit_length = max_hit_length
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._fact_key = fact_key
        self._use_top_relation = use_top_relation
        self._ignore_spans = ignore_spans
        self._aggressive_filtering = aggressive_filtering
        self._add_relation_labels = add_relation_labels
        self._max_tuples = max_tuples
        self._logged = False

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as data_file:
            logger.info("Reading QA + fact instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                item_json = json.loads(line.strip())

                item_id = item_json["id"]
                question_text = item_json["question"]["stem"]
                fact = item_json[self._fact_key]
                answer_spans = item_json["answer_spans"]
                answer_starts = item_json.get("answer_starts", None)
                answer_relations = item_json.get("relations", None)
                choice_label_to_id = {}
                choice_text_list = []
                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id
                    choice_text = choice_item["text"]
                    choice_text_list.append(choice_text)

                answer_id = choice_label_to_id[item_json["answerKey"]]
                prefetched_sentences = item_json.get("prefetched_sentences", None)
                prefetched_indices = item_json.get("prefetched_indices", None)
                yield self.text_to_instance(item_id, question_text, choice_text_list, fact,
                                            answer_spans, answer_relations, answer_starts,
                                            answer_id, prefetched_sentences, prefetched_indices)

    def collate_relations(self, relations: List[str]) -> List[str]:
        """
        Return a list of unique relations from all the annotated relations
        """
        relation_counts = Counter(relations)
        most_common_count = relation_counts.most_common(1)[0][1]

        if most_common_count <= 1 and self._use_top_relation:
            # No repeated relation
            return ["related"]
        else:
            if not self._use_top_relation:
                return list(relation_counts.keys()) + ["related"]
            else:
                relation_list = []
                for k, v in relation_counts.most_common():
                    if v < most_common_count:
                        break
                    else:
                        relation_list.append(k)
                return relation_list

    def get_elasticsearch_sentences(self, prefetched_sentences: Dict[str, List[str]],
                                    prefetched_indices: str,
                                    answer_span: List[str], choice: str,
                                    question: str, fact: str, max_hits: int) -> List[str]:
        if (prefetched_sentences is None or
            prefetched_indices is None or
             prefetched_indices != self._indices):
            if not self._logged:
                logger.info("Retrieving sentences")
                self._logged = True
            choice_hits = []
            if self._ignore_spans:
                fact_hits = self._es_search.get_hits_for_question(
                    question=fact, choices=[choice])[choice]
                choice_hits.extend(fact_hits)
            else:
                for answer in set(answer_span):
                    answer_choice_hits = self._es_search.get_hits_for_question(
                        question=answer, choices=[choice])[choice]
                    choice_hits.extend(answer_choice_hits)
            choice_hits.sort(key=lambda x: -x.score)
            filtered_sentences = []
            for es_hit in choice_hits:
                if es_hit.text not in filtered_sentences:
                    filtered_sentences.append(es_hit.text)
        else:
            if not self._logged:
                logger.info("Using pre-retrieved sentences")
                self._logged = True
            filtered_sentences = [sentence for sentence in prefetched_sentences[choice]
                                  if len(sentence) <= self._max_hit_length]
        if self._aggressive_filtering:
            filtered_sentences = self.filter_near_duplicates(filtered_sentences)

        topk_unique_hit_texts = filtered_sentences[:max_hits]

        return topk_unique_hit_texts

    def filter_near_duplicates(self, sentences: List[str]) -> List[str]:
        """Filters out items that overlap too much with items we've seen earlier in the sequence."""
        trigram_to_sentence_indices = {}
        result = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_tokens = [stem for (token, stem) in tokenize_and_stem_str(sentence) if stem]
            trigrams = [sentence_tokens[i:i + 3] for i in range(len(sentence_tokens) - 3)]
            if len(trigrams) <= 1:
                # too small, ignore
                continue
            overlapping_sentence_indices = Counter()
            for trigram in trigrams:
                trigram_key = " ".join(trigram)
                for si in trigram_to_sentence_indices.get(trigram_key, []):
                    overlapping_sentence_indices[si] += 1

            if len(overlapping_sentence_indices) > 0:
                max_overlap = max(overlapping_sentence_indices.values())
            else:
                max_overlap = 0
            # high overlap
            if max_overlap / len(trigrams) >= 0.9:
                continue
            # okay to add
            result.append(sentence)
            for trigram in trigrams:
                trigram_key = " ".join(trigram)
                if trigram_key not in trigram_to_sentence_indices:
                    trigram_to_sentence_indices[trigram_key] = []
                trigram_to_sentence_indices[trigram_key].append(sentence_index)
        return result

    def get_conceptnet_sentences(self, fact: str, answer_span: List[str],
                                 choice: str, max_tuples: int):
        choice_tuples = []
        if self._ignore_spans:
            fact_choice_tuples = retrieve_scored_tuples(fact, choice, self._kb_tuples,
                                                        self._tok_idx_map, max_tuples)
            choice_tuples.extend(fact_choice_tuples)
        else:
            for answer in set(answer_span):
                answer_choice_tuples = retrieve_scored_tuples(answer, choice, self._kb_tuples,
                                                              self._tok_idx_map, max_tuples)
                choice_tuples.extend(answer_choice_tuples)
        choice_tuples.sort(key=lambda x: -x[1])
        topk_unique_tuples = []
        for tuple, score in choice_tuples:
            if tuple not in topk_unique_tuples:
                topk_unique_tuples.append(tuple)
            if len(topk_unique_tuples) >= max_tuples:
                break
        topk_tuple_sentences = []
        for tuple in topk_unique_tuples:
            updated_relation = convert_relation_to_string(tuple[1])
            tuple_as_sentence = convert_entity_to_string(tuple[0]) + " is " + updated_relation + \
                                " " + convert_entity_to_string(tuple[2])
            topk_tuple_sentences.append(tuple_as_sentence)
        return topk_tuple_sentences

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: Any,
                         question_text: str,
                         choice_text_list: List[str],
                         fact_text: str,
                         answer_span: List[str],
                         answer_relations: List[str],
                         answer_starts: List[int] = None,
                         answer_id: int = None,
                         prefetched_sentences: Dict[str, List[str]] = None,
                         prefetched_indices: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question_text)
        fact_tokens = self._tokenizer.tokenize(fact_text)
        choices_tokens_list = [self._tokenizer.tokenize(x) for x in choice_text_list]
        choice_kb_fields = []
        selected_tuples = []
        for choice in choice_text_list:
            kb_fields = []

            if self._use_conceptnet and self._use_elastic_search:
                max_sents_per_source = int(self._max_tuples / 2)
            else:
                max_sents_per_source = self._max_tuples
            selected_hits = []
            if self._use_elastic_search:
                elastic_search_hits = self.get_elasticsearch_sentences(prefetched_sentences,
                                                                       prefetched_indices,
                                                                       answer_span, choice,
                                                                       question_text, fact_text,
                                                                       max_sents_per_source)
                selected_hits.extend(elastic_search_hits)

            if self._use_conceptnet:
                conceptnet_sentences = self.get_conceptnet_sentences(fact_text, answer_span, choice,
                                                                     max_sents_per_source)
                selected_hits.extend(conceptnet_sentences)
            # add a dummy entry to capture the embedding link
            if self._ignore_spans:
                fact_choice_sentence = fact_text + " || " + choice
                selected_hits.append(fact_choice_sentence)
            else:
                for answer in set(answer_span):
                    answer_choice_sentence = answer + " || " + choice
                    selected_hits.append(answer_choice_sentence)

            selected_tuples.append(selected_hits)
            for hit_text in selected_hits:
                kb_fields.append(TextField(self._tokenizer.tokenize(hit_text),
                                           self._token_indexers))

            choice_kb_fields.append(ListField(kb_fields))

        fields["choice_kb"] = ListField(choice_kb_fields)
        fields['fact'] = TextField(fact_tokens, self._token_indexers)

        if self._add_relation_labels:
            if answer_relations and len(answer_relations):
                relation_fields = []
                for relation in set(answer_relations):
                    relation_fields.append(LabelField(relation, label_namespace="relation_labels"))
                fields["relations"] = ListField(relation_fields)
                selected_relations = self.collate_relations(answer_relations)
                fields["relation_label"] = MultiLabelField(selected_relations, "relation_labels")
            else:
                fields["relations"] = ListField([LabelField(-1, label_namespace="relation_labels",
                                                            skip_indexing=True)])
                fields["relation_label"] = MultiLabelField([], "relation_labels")

        answer_fields = []
        answer_span_fields = []
        fact_offsets = [(token.idx, token.idx + len(token.text)) for token in fact_tokens]

        for idx, answer in enumerate(answer_span):
            answer_fields.append(TextField(self._tokenizer.tokenize(answer),
                                           self._token_indexers))
            if answer_starts:
                if len(answer_starts) <= idx:
                    raise ValueError("Only {} answer_starts in json. "
                                     "Expected {} in {}".format(len(answer_starts),
                                                                len(answer_span),
                                                                item_id))
                offset = answer_starts[idx]
            else:
                offset = fact_text.index(answer)
                if offset == -1:
                    raise ValueError("Span: {} not found in fact: {}".format(answer, fact_text))

            tok_span, err = char_span_to_token_span(fact_offsets, (offset, offset + len(answer)))
            if err:
                logger.info("Could not find token spans for '{}' in '{}'."
                            "Best guess: {} in {} at {}".format(
                    answer, fact_text, [offset, offset + len(answer)], fact_offsets, tok_span))
            answer_span_fields.append(SpanField(tok_span[0], tok_span[1], fields['fact']))

        fields["answer_text"] = ListField(answer_fields)
        fields["answer_spans"] = ListField(answer_span_fields)
        fields['question'] = TextField(question_tokens, self._token_indexers)

        fields['choices_list'] = ListField(
            [TextField(x, self._token_indexers) for x in choices_tokens_list])
        if answer_id is not None:
            fields['answer_id'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "fact_text": fact_text,
            "choice_text_list": choice_text_list,
            "question_tokens": [x.text for x in question_tokens],
            "fact_tokens": [x.text for x in fact_tokens],
            "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
            "answer_text": answer_span,
            "answer_start": answer_starts,
            "answer_span_fields": [(x.span_start, x.span_end) for x in answer_span_fields],
            "relations": answer_relations,
            "selected_tuples": selected_tuples
        }

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

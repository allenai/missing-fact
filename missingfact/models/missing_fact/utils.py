import torch
from allennlp.nn.util import get_text_field_mask
from torch.nn.functional import sigmoid

from missingfact.nn.util import seq2vec_seq_aggregate



def add_tuple_predictions(kb_relation_label_logits, metadata):
    for idx, batch in enumerate(kb_relation_label_logits):
        if "debug_info" not in metadata[idx]:
            metadata[idx]["debug_info"] = {}
        metadata[idx]["debug_info"]["tuple_relation_probs"] = []
        metadata[idx]["debug_info"]["tuple_labels"] = []
        for cidx, choice in enumerate(batch):
            formatted_tuples = []
            for x in metadata[idx]['selected_tuples'][cidx]:
                if isinstance(x, list):
                    formatted_tuples.append(" | ".join(x))
                else:
                    formatted_tuples.append(x)
            metadata[idx]["debug_info"]["tuple_labels"].append(formatted_tuples)
            num_tuples = len(metadata[idx]["debug_info"]["tuple_labels"][-1])
            metadata[idx]["debug_info"]["tuple_relation_probs"].append(
                torch.sigmoid(kb_relation_label_logits[idx, cidx, 0:num_tuples, :]).detach().cpu().numpy())

def add_relation_predictions(vocab, relation_label_prob_tensor, metadata):
    # output_relations = []
    relation_label_probs = relation_label_prob_tensor.detach().cpu().numpy()
    selected_relation = torch.argmax(relation_label_prob_tensor, 2).detach().cpu().numpy()
    row_labels = []
    for rel_idx in range(vocab.get_vocab_size(namespace="relation_labels")):
        row_labels.append(vocab.get_token_from_index(rel_idx,
                                                     namespace="relation_labels"))
    for idx, batch in enumerate(selected_relation):
        batch_relations = []
        for cidx, choice in enumerate(batch):
            rel = vocab.get_token_from_index(choice.item(),
                                             namespace="relation_labels")
            ridx = choice.item()
            if rel == "related":
                topk, indices = torch.topk(relation_label_prob_tensor[idx, cidx, :], 2)
                rel = vocab.get_token_from_index(indices[1].item(),
                                                 namespace="relation_labels")
                ridx = indices[1].item()
            batch_relations.append((rel, relation_label_probs[idx, cidx, ridx]))
        # output_relations.append(batch_relations)
        if "debug_info" not in metadata[idx]:
            metadata[idx]["debug_info"] = {}
        metadata[idx]["debug_info"]["relation_labels"] = row_labels
        metadata[idx]["debug_info"]["choice_labels"] = metadata[idx]["choice_text_list"]
        metadata[idx]["debug_info"]["relation_probs"] = relation_label_probs[idx, :, :]
        metadata[idx]["debug_info"]["relation_predictions"] = batch_relations


def get_text_representation(text_tensor, num_wrapping_dims, embedder, encoder, aggregate):
    embedded_text, text_mask = get_embedding(text_tensor, num_wrapping_dims, embedder, encoder)
    return get_agg_rep(embedded_text, text_mask, num_wrapping_dims, encoder, aggregate)


def get_embedding(text_tensor, num_wrapping_dims, embedder, encoder, var_dropout=None):
    if num_wrapping_dims > 0:
        squashed_text = {}
        squashed_dimensions = None
        for key, tensor in text_tensor.items():
            squashed_shape = [-1] + [x for x in tensor.size()[num_wrapping_dims + 1:]]
            squashed_tensor = tensor.contiguous().view(*squashed_shape)
            squashed_text[key] = squashed_tensor
            if key == "tokens":
                # print(squashed_tensor.size())
                squashed_dimensions = tensor.size()[:num_wrapping_dims + 2]
        embedded_text = embedder(squashed_text)
        text_mask = get_text_field_mask(squashed_text).float()
        if var_dropout is not None:
            embedded_text = var_dropout(embedded_text)
        if encoder:
            embedded_text = encoder(embedded_text, text_mask)

        mask_shape = [x for x in squashed_dimensions]
        output_size = mask_shape + [-1]
        return embedded_text.contiguous().view(*output_size), \
               text_mask.contiguous().view(*mask_shape)
    else:
        embedded_text = embedder(text_tensor, num_wrapping_dims=num_wrapping_dims)
        if var_dropout is not None:
            embedded_text = var_dropout(embedded_text)
        # print(embedded_text.shape)
        text_mask = get_text_field_mask(text_tensor, num_wrapping_dims).float()
        # print(text_mask.shape)
        if encoder:
            embedded_text = encoder(embedded_text, text_mask)
        return embedded_text, text_mask


def get_agg_rep(embedded_text, text_mask, num_wrapping_dims, encoder, aggregate):
    return seq2vec_seq_aggregate(embedded_text,
                                 text_mask,
                                 aggregate,
                                 encoder and encoder.is_bidirectional(),
                                 num_wrapping_dims + 1), \
           (torch.sum(text_mask, dim=num_wrapping_dims + 1) != 0).float()

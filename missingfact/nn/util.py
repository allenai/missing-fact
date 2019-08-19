import torch
from allennlp.nn.util import replace_masked_values, masked_max


def seq2vec_seq_aggregate(seq_tensor, mask, aggregate, bidirectional, dim=1):
    """
        Takes the aggregation of sequence tensor

        :param seq_tensor: Batched sequence requires [batch, seq, hs]
        :param mask: binary mask with shape batch, seq_len, 1
        :param aggregate: max, avg, sum
        :param dim: The dimension to take the max. for batch, seq, hs it is 1
        :return:
    """

    seq_tensor_masked = seq_tensor * mask.unsqueeze(-1)
    aggr_func = None
    if aggregate == "last":
        if seq_tensor.dim() > 3:
            seq = get_final_encoder_states_after_squashing(seq_tensor, mask, bidirectional)
        else:
            seq = get_final_encoder_states(seq_tensor, mask, bidirectional)
    elif aggregate == "max":
        seq = masked_max(seq_tensor, mask.unsqueeze(-1).expand_as(seq_tensor), dim=dim)
    elif aggregate == "min":
        seq = -masked_max(-seq_tensor, mask.unsqueeze(-1).expand_as(seq_tensor), dim=dim)
    elif aggregate == "sum":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "avg":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
        seq_lens = torch.sum(mask, dim=dim)  # this returns batch_size, .. 1 ..
        masked_seq_lens = replace_masked_values(seq_lens, (seq_lens != 0).float(), 1.0)
        masked_seq_lens = masked_seq_lens.unsqueeze(dim=dim).expand_as(seq)
        # print(seq.shape)
        # print(masked_seq_lens.shape)
        seq = seq / masked_seq_lens

    return seq


def get_final_encoder_states_after_squashing(embedded_text, text_mask, bidirectional):
    # print(embedded_text.size())
    squashed_shape = [-1, embedded_text.size()[-2], embedded_text.size()[-1]]
    # print(squashed_shape)
    squashed_text = embedded_text.contiguous().view(*squashed_shape)
    squash_mask_shape = [squashed_text.size()[0], squashed_text.size()[1]]
    squashed_mask = text_mask.contiguous().view(*squash_mask_shape)
    squashed_final_seq = get_final_encoder_states(squashed_text, squashed_mask, bidirectional)
    # print(squashed_final_seq.size())
    output_size = [x for x in embedded_text.size()[:-2]] + [-1]
    return squashed_final_seq.contiguous().view(*output_size)


def get_final_encoder_states(encoder_outputs: torch.Tensor,
                             mask: torch.Tensor,
                             bidirectional: bool = False) -> torch.Tensor:
    """
    Modified over the original Allennlp function

    Given the output from a ``Seq2SeqEncoder``, with shape ``(batch_size, sequence_length,
    encoding_dim)``, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape ``(batch_size, encoding_dim)``.  This is not as simple as
    ``encoder_outputs[:, -1]``, because the sequences could have different lengths.  We use the
    mask (which has shape ``(batch_size, sequence_length)``) to find the final state for each batch
    instance.

    Additionally, if ``bidirectional`` is ``True``, we will split the final dimension of the
    ``encoder_outputs`` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving ``encoder_outputs[:, -1, :encoding_dim/2]`` concated with
    ``encoder_outputs[:, 0, encoding_dim/2:]``.
    """
    # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
    # are assuming sequences are right padded.
    # Shape: (batch_size,)
    last_word_indices = mask.sum(1).long() - 1

    # handle -1 cases
    ll_ = (last_word_indices != -1).long()
    last_word_indices = last_word_indices * ll_

    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    # Shape: (batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2):]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output
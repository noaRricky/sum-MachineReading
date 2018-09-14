import torch
import torch.nn as nn
import numpy as np

from .constants import PAD_TOKEN
from .blocks import RecurrentEncoderBlock, RecurrentDecoderBlock


def position_encoding_init(n_position, d_pos_vec):
    """ Init the sinusoid position encoding table

    Args:
        n_position (int): number of the postion
        d_pos_vec (int): dimension of the position encoding vector
    """
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec)
         for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)
    ])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[0:, 0::2] = np.cos(position_enc[0:, 0::2])

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def time_encoding_init(n_time, d_time_vec):
    """ Init the sinusoid time encoding table

    Args:
        n_time (int): number the attention time
        d_time_vec (int): dimension of time encoding vector
    """
    time_enc = np.array([
        [time / np.power(10000, 2 * (j // 2) / d_time_vec)
         for j in range(d_time_vec)]
        if time != 0 else np.zeros(d_time_vec) for time in range(n_time)
    ])

    time_enc[1:, 0::2] = np.sin(time_enc[1:, 0::2])
    time_enc[0:, 0::2] = np.cos(time_enc[0:, 0::2])

    return torch.from_numpy(time_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):
    """ Indicate the padding-related part to mask

    Args:
        seq_q (tensor): query tensor, shape (batch, len)
        seq_k (tensor): key tensors, shape (batch, len)

    Returns:
        pad_attn_mask: mask for query and key, shape (batch, len_q, len_k)
    """

    assert seq_q.dim() == 2 and seq_k.dim() == 2

    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()

    # shape (mb_size, 1, len_k)
    pad_attn_mask = seq_k.data.eq(PAD_TOKEN).unsqueeze(1)
    # shape (mb_size, len_q, len_k)
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)

    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    """ Get an attention mask to avoid using the subsequent info

    Args:
        seq (tensor): shape (batch, len_s)

    Returns:
        subsequent_mask: mask tensor for each batch, shape (batch, len_s, len_s)
    """

    assert seq.dim() == 2

    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask).astype(seq)
    return subsequent_mask


class RecurrentEncoder(nn.Module):

    def __init__(self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, dropout_rate=0.1):
        super(RecurrentEncoder, self).__init__()

        n_position = n_max_seq + 1
        n_time = n_layers
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        # create positon encoder
        self.position_encoder = nn.Embedding(
            n_position, d_word_vec, padding_idx=PAD_TOKEN)
        self.position_encoder.weight.data = position_encoding_init(
            n_position, d_word_vec)

        # create time encoder
        self.time_encoder = nn.Embedding(
            n_time, d_word_vec, padding_idx=PAD_TOKEN)
        self.time_encoder.weight.data = time_encoding_init(n_time, d_word_vec)

        # create src embedding
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=PAD_TOKEN)

        self.layer_stack = nn.ModuleList([
            RecurrentEncoderBlock(d_model, d_inner_hid, n_head, d_k, d_v) 
            for _ in range(n_layers)])

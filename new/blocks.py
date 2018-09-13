import torch
import torch.nn as nn
import numpy as np

from .sublayers import MultiHeadAttention, PositionWiseFeedForward


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


class RecurrentEncoderBlock(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout_rate=0.1):
        super(RecurrentEncoderBlock, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_normalize1 = nn.LayerNorm(d_model)
        self.layer_normalize2 = nn.LayerNorm(d_model)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner_hid)

    def forward(self, enc_input, enc_pos, enc_time, position_encoder, time_encoder, self_attn_mask=None):
        
        # Position Encoding addition
        enc_input += position_encoder(enc_pos)

        # Time encoding addition
        enc_input += time_encoder(enc_time)

        # apply multi-attention operation and residual
        enc_residual = enc_input
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output += enc_residual

        # firt dropout and layer normalization
        enc_output = self.dropout1(enc_output)
        enc_output = self.layer_normalize1(enc_output)

        # apply position wise feed forward and residual
        enc_residual = enc_output
        enc_output = self.pos_ffn(enc_output)
        enc_output += enc_residual

        # dropout and layer normalization operation
        enc_output = self.dropout2(enc_output)
        enc_output = self.layer_normalize2(enc_output)

        return enc_output, enc_slf_attn


class RecurrentDecoderBlock(nn.Module):
    
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout_rate=0.1):
        pass

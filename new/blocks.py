import torch.nn as nn

from .sublayers import MultiHeadAttention, PositionWiseFeedForward


class RecurrentEncoderBlock(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v):
        super(RecurrentEncoderBlock, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.transition_fn = PositionWiseFeedForward(d_model, d_inner_hid)

    def forward(self, enc_input, enc_pos, enc_time, position_encoder, time_encoder, self_attn_mask=None):

        # Position Encoding addition
        enc_input += position_encoder(enc_pos)

        # Time encoding addition
        enc_input += time_encoder(enc_time)

        # apply multi-attention operation
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)

        # apply position wise feed forward and residual
        enc_output = self.transition_fn(enc_output)

        return enc_output, enc_slf_attn


class RecurrentDecoderBlock(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v):
        super(RecurrentDecoderBlock, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.tran_fn = PositionWiseFeedForward(d_model, d_inner_hid)

    def forward(self, dec_input, enc_output, dec_pos, dec_time, pos_encoder, time_encoder, slf_attn_mask=None, dec_enc_attn_mask=None):

        # Position decoding addition
        dec_input += pos_encoder(dec_pos)

        # Time decoding addidtion
        dec_input += time_encoder(dec_time)

        # first self attention and residual
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)

        # multihead attention
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)

        # transition function and residual
        dec_output = self.tran_fn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn

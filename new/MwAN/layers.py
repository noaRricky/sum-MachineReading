import torch
import torch.nn as nn
import torch.nn.functional as F

from attens import MultiHeadAtten


class PositionwiseFeedForward(nn.Module):
    """ a　two layer feed forward"""

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(residual + output)
        return output


class TransformerEncoder(nn.Module):
    """ Transformer encoder """

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.1)
    super(TransformerEncoder, self).__init__()

    self.attention = MultiHeadAtten(model_dim, num_heads, dropout)
    self.feed_forward = PositionwiseFeedForward(
        model_dim, ffn_dim, dropout)

    def forward(self, inputs, atten_mask=None):

        # self attention
        context, atten = self.attention(inputs, inputs, inputs, atten_mask)
        # feed forward
        output = self.feed_forward(context)
        return output, atten

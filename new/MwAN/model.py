import torch
import torch.nn as nn

from layers import EncodeLayer
from attens import ConcatAtten, BilinearAtten, DotAtten, MinusAtten


class MwANP(nn.Module):

    def __init__(self, vocab_size, embed_size, encode_size, num_heads=8,
                 ffn_dim=2018, dropout=0.1):
        super(MwANP, self).__init__()

        # embedding part
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # encode part
        self.ans_encoder = EncodeLayer(
            embed_size, encode_size, num_heads, ffn_dim, dropout)
        self.que_encoder = EncodeLayer(
            embed_size, encode_size, num_heads, ffn_dim, dropout)
        self.pas_encoder = EncodeLayer(
            embed_size, encode_size, num_heads, ffn_dim, dropout)

        # attention part
        # mult-head concat attention

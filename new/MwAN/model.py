import torch
import torch.nn as nn

from layers import EncodeLayer, AggregateLayer, PredictionLayer
from attens import ConcatAtten, BilinearAtten, DotAtten, MinusAtten, MultiHeadAtten


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
        self.mult_concat_atten = MultiHeadAtten(
            ConcatAtten, encode_size, num_heads, dropout)
        # multi-head bilinear attention
        self.multi_bilinear_atten = MultiHeadAtten(
            BilinearAtten, encode_size, num_heads, dropout)
        # multi-head dotatten attention
        self.multi_dot_atten = MultiHeadAtten(
            DotAtten, encode_size, num_heads, dropout)
        # multi-head minus attention
        self.multi_minus_atten = MultiHeadAtten(
            MinusAtten, encode_size, num_heads, dropout)

        # aggregation part
        self.agg_layer = AggregateLayer(encode_size)

        # predict layer
        self.predict_layer = PredictionLayer(encode_size, dropout)

    def forward(self, passage, question, answer, is_train):

        # embedding layer
        pas_embed = self.embed_dropout(self.embedding(passage))
        que_embed = self.embed_dropout(self.embedding(question))
        ans_embed = self.embed_dropout(self.embedding(answer))

        # encode layer
        pas_encode = self.pas_encoder(pas_embed)
        que_encode = self.que_encoder(que_embed)
        ans_encode = self.ans_encoder(ans_embed)

        # attention layer
        qtc = self.mult_concat_atten(pas_encode, que_encode, que_encode)
        qtb = self.multi_bilinear_atten(pas_encode, que_encode, que_encode)
        qtd = self.multi_dot_atten(pas_encode, que_encode, que_encode)
        qtm = self.multi_minus_atten(pas_encode, que_encode, que_encode)

        # aggregation layer
        agg = self.agg_layer(qtc, qtb, qtd, qtm)

        # prediction layer
        score = self.predict_layer(que_encode, ans_encode, agg)

        if not is_train:
            return score.argmax(1)
        else:
            loss = -torch.log(score[:, 0]).mean()
            return loss

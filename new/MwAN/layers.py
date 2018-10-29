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

    def __init__(self, encode_size=512, num_heads=8, ffn_dim=2018, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.attention = MultiHeadAtten(encode_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(
            encode_size, ffn_dim, dropout)

    def forward(self, inputs, atten_mask=None):

        # self attention
        context, atten = self.attention(inputs, inputs, inputs, atten_mask)
        # feed forward
        output = self.feed_forward(context)
        return output, atten


class RNMTPlusEncoder(nn.Module):

    def __init__(self, model_dim, dropout=0.1):
        super(RNMTPlusEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.lstm = nn.LSTM(model_dim, model_dim,
                            batch_first=True, bidirectional=True)

    def forward(self, inputs):

        residual = inputs
        output, _ = self.lstm(inputs)
        output = self.dropout(output)
        return self.layer_norm(residual + output)


class AggregateLayer(nn.Module):

    def __init__(self, model_dim):
        super(AggregateLayer, self).__init__()

        self.gru_agg(6 * model_dim, model_dim,
                     batch_first=True, bidirectional=True)

    def forward(self, hp, qts, qtc, qtd, qtb, qtm):
        aggregation = torch.cat([hp, qts, qtc, qtd, qtb, qtm], dim=2)
        aggregation_represent, _ = self.gru_agg(aggregation)


class PredictionLayer(nn.Module):

    def __init__(self, model_dim, dropout=0.1):
        super(PredictionLayer, self).__init__()

        self.Wq = nn.Linear(2 * model_dim, model_dim, bias=False)
        self.vq = nn.Linear(model_dim, 1, bias=False)
        self.Wp1 = nn.Linear(2 * model_dim, model_dim, bias=False)
        self.Wp2 = nn.Linear(2 * model_dim, model_dim, bias=False)
        self.vp = nn.Linear(model_dim, 1, bias=False)
        self.prediction = nn.Linear(2 * model_dim, model_dim, bias=False)
        self.que_softmax = nn.Softmax(dim=2)
        self.agg_softmax = nn.Softmax(dim=2)
        self.score_softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_question, ans_embed, agg_represent):
        sj = self.vq(torch.tanh(self.Wq(hidden_question))).transpose(2, 1)
        rq = self.que_softmax(sj).bmm(hidden_question)
        sj = self.agg_softmax(self.vp(self.Wp1(agg_represent) + self.Wp2(rq)))
        rp = torch.bmm(sj, agg_represent)
        output = self.dropout(F.leaky_relu(self.prediction(rp)))
        score = self.score_softmax(
            torch.bmm(ans_embed, output.transpose(2, 1)).squeeze())
        return score

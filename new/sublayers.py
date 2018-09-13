import torch
import torch.nn as nn
import torch.nn.init as init


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    """

    def __init__(self, d_k, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scale_factor = 1 / (d_k ** 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask=None):
        """compute the similarity of q and v by attention machinesim

        Arguments:
            q {tensor} -- shape '(batch, n, d_k)'
            k {tensor} -- shape '(batch, m, d_k)'
            v {tensor} -- shape '(batch, m, d_v)'

        Keyword Arguments:
            atten_mask {tensor} -- shape '(batch, n, m)' (default: {None})

        Returns:
            output, attn -- output refer to attention scaled
            result for values, attn is the attention result
        """

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale_factor

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                "Attention mask shape {} mismatch " \
                "with Attention logit tensor shape " \
                "{}.".format(attn_mask.size(), attn.size())

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """MultiHead attention model

    Arguments:
        n_head {int} -- number of head
        d_model {int} -- hidden size of model
        d_k {int} -- dimension of key
        d_v {int} -- dimension of value
        dropout_rate {float} -- set for dropout operation
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.weight_qs = nn.Parameter(torch.Tensor(n_head, d_model, d_k))
        self.weight_ks = nn.Parameter(torch.Tensor(n_head, d_model, d_k))
        self.weight_vs = nn.Parameter(torch.Tensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        init.xavier_normal_(self.weight_qs)
        init.xavier_normal_(self.weight_ks)
        init.xavier_normal_(self.weight_vs)

    def forward(self, q, k, v, attn_mask=None):

        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

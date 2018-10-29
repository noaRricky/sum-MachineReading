import numpy as np
import torch
import torch.nn as nn


class MultiHeadAtten(nn.Module):
    """Multi head attetnion"""

    def __init__(self, atten_unit, encode_size, num_heads=8, dropout=0.1):
        super(MultiHeadAtten, self).__init__()

        model_dim = encode_size * 2

        self.attention = atten_unit
        self.num_heads = num_heads
        self.dim_per_head = d_k = d_v = model_dim // num_heads

        self.linear_q = nn.Linear(model_dim, num_heads * d_k)
        self.linear_k = nn.Linear(model_dim, num_heads * d_k)
        self.linear_v = nn.Linear(model_dim, num_heads * d_v)

        unit_encode_size = d_k // 2
        self.attention = atten_unit(unit_encode_size, dropout)
        self.linear_final = nn.Linear(num_heads * d_v, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, atten_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = query.size(0)

        # linear projectin
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # split by heads
        query = query.view(batch_size * num_heads, -1, dim_per_head)
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)

        if atten_mask:
            atten_mask = atten_mask.repeat(num_heads, 1, 1)

        # scaled dot product attention
        context, atten = self.attention(query, key, value, atten_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, atten


class ScaledDotProductAtten(nn.Module):
    """Scaled dot-product attention mechainsm
    """

    def __init__(self, encode_size, atten_dropout=0.1):
        super(ScaledDotProductAtten, self).__init__()

        encode_size = 2 * encode_size
        self.scale = encode_size ** -0.5
        self.dropout = nn.Dropout(atten_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, atten_mask=None):
        """前向传播.

        Args:
                q: Queries张量，形状为[B, L_q, D_q]
                k: Keys张量，形状为[B, L_k, D_k]
                v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
                scale: 缩放因子，一个浮点标量
                attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
                上下文张量和attetention张量
        """
        atten = torch.bmm(query, key.transpose(1, 2)) * self.scale
        if atten_mask:
            # 给需要mask的地方设置一个负无穷
            atten.masked_fill_(atten_mask, -np.inf)
        atten = self.softmax(atten)
        # 添加dropout
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class ConcatAtten(nn.Module):

    def __init__(self, encode_size, atten_dropout=0.0):
        super(ConcatAtten, self).__init__()

        self.Wc1 = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.Wc2 = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.vc = nn.Linear(encode_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value, atten_mask=None):
        q = self.Wc1(query).unsqueeze(1)
        k = self.Wc2(key).unsqueeze(2)
        sjt = self.vc(torch.tanh(q + k)).squeeze()

        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)

        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class BilinearAtten(nn.Module):

    def __init__(self, encode_size: int, atten_dropout=0.0):
        super(BilinearAtten, self).__init__()

        self.Wb = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value, atten_mask=None):
        s1 = self.Wb(query).transpose(2, 1)
        sjt = torch.bmm(key, s1)
        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)
        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(value, atten)
        return context, atten


class DotAtten(nn.Module):

    def __init__(self, encode_size: int, atten_dropout=0.0):
        super(DotAtten, self).__init__()

        self.Wd = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.vd = nn.Linear(encode_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value, atten_mask=None):
        q = query.unsqueeze(1)
        k = query.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(q * k))).squeeze()

        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)

        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class MinusAtten(nn.Module):

    def __init__(self, encode_size, atten_dropout=0.1):
        super(MinusAtten, self).__init__()

        self.Wm = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.vm = nn.Linear(encode_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value, atten_mask=None):

        q = query.unsqueeze(1)
        k = query.unsqueeze(2)
        sjt = self.vm(torch.tanh(self.Wm(q - k))).squeeze()

        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)

        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


"""
class SelfAtten(nn.Module):

    def __init__(self, model_dim, atten_dropout=0.1):
        super(SelfAtten, self).__init__()

        self.Ws = nn.Linear(2 * model_dim, model_dim, bias=False)
        self.vs = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(atten_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, hidden, atten_mask=None):

        query = hidden.unsqueeze(1)
        key = hidden.unsqueeze(2)
        value = hidden

        sjt = self.vs(torch.tanh(self.Ws(query * key))).squeeze()

        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)

        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten
"""

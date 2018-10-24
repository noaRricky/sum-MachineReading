import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAtten(nn.Module):
    """Scaled dot-product attention mechainsm
    """

    def __init__(self, scale=None, atten_drop=0.1):
        super(ScaledDotProductAtten, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(atten_drop)
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
        atten = torch.bmm(query, key.transpose(1, 2))
        if atten_mask:
            # 给需要mask的地方设置一个负无穷
            atten.masked_fill_(atten_mask, -np.inf)
        atten = self.softmax(atten)
        # 添加dropout
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class ConcatAtten(nn.Module):

    def __init__(self, encoder_size: int):
        super(ConcatAtten, self).__init__()

        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)

    def forward(self, query, key, value):
        q = self.Wc1(query).unsqueeze(1)
        k = self.Wc2(key).unsqueeze(2)
        sjt = self.vc(torch.tanh(q + k)).squeeze()
        attens = torch.softmax(sjt, 2)
        attens = torch.bmm(value, attens)
        return attens


class BilinearAtten(nn.Module):

    def __init__(self, encoder_size: int):
        super(BilinearAtten, self).__init__()

        self.Wb = nn.Linear(2 * encoder_size, encoder_size, bias=False)

    def forward(self, query, key, value):
        q = self.Wb(query).transpose(2, 1)
        attens = torch.bmm(key, q)
        attens = torch.softmax(attens, 2)
        attens = torch.bmm(value, attens)
        return attens


class DotAtten(nn.Module):

    def __init__(self, encoder_size: int):
        super(DotAtten, self).__init__()

        self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)

    def forward(self, query, key, value):
        q = query.unsqueeze(1)
        k = query.unsqueeze(2)
        attens = self.vd(torch.tanh(self.Wd(q, k))).squeeze()
        attens = self.softmax(attens, dim=2)

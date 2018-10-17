import torch
import torch.nn as nn


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

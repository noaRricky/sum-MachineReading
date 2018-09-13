import torch
import torch.nn as nn
import torch.nn.init as init


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention

    Arguments:
        d_k {int} -- dimension of key
        attn_dropout {float} -- dropout rate for attention result
    """

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()

        self.scale_factor = 1 / (d_k ** 0.5)
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
            output, attn -- output shape (batch, n, d_v) 
            attn shape (batch, n, m)
        """

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale_factor

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                "Attention mask shape {} mismatch " \
                "with Attention logit tensor shape " \
                "{}.".format(attn_mask.size(), attn.size())

        attn = self.softmax(attn)
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

    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.weight_qs = nn.Parameter(torch.Tensor(n_head, d_model, d_k))
        self.weight_ks = nn.Parameter(torch.Tensor(n_head, d_model, d_k))
        self.weight_vs = nn.Parameter(torch.Tensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, dropout_rate)
        self.proj = nn.Linear(n_head * d_v, d_model)

        init.xavier_normal_(self.weight_qs)
        init.xavier_normal_(self.weight_ks)
        init.xavier_normal_(self.weight_vs)
        init.xavier_normal_(self.proj.weight)

    def forward(self, q, k, v, attn_mask=None):
        """whole forward multi head attention opertaion without residual function

        Arguments:
            q {tensor} -- shpae (batch, len_q, d_model)
            k {tensor} -- shape (batch, len_k, d_model)
            v {tensor} -- shape (batch, len_v, d_model)

        Keyword Arguments:
            atten_mask {tenor} -- shape (batch, n, m) used for decoder (default: {None})
        """

        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        # shape (n_head, mb_size * len_q, d_model)
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        # shape (n_head, mb_size * len_k, d_model)
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        # shape (n_head, mb_size * len_v, d_model)
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        # treat the result as a (n_head * mb_size) size batch
        # shape (n_head * mb_size, len_q, d_k)
        q_s = torch.bmm(q_s, self.weight_qs).view(-1, len_q, d_k)
        # shape (n_head * mb_size, len_k, d_k)
        k_s = torch.bmm(k_s, self.weight_ks).view(-1, len_k, d_k)
        # shape (n_head * mb_size, len_v, d_v)
        v_s = torch.bmm(v_s, self.weight_vs).view(-1, len_v, d_v)

        # perform scale-dot attention, output shape (n_head * mb_size, len_q, d_v)
        outputs, attns = self.attention(
            q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))

        # back to original mb_size batch
        # split result shape for each (mb_size, len_q, d_v) with n_head amunt
        outputs = torch.split(outputs, mb_size, dim=0)
        # concat result shape (mb_size, len_q, n_head * d_v)
        outputs = torch.cat(outputs, dim=2)

        # preject back to d_model size
        outputs = self.proj(outputs)

        return outputs, attns


class PositionWiseFeedForward(nn.Module):
    """ A two feed forward layer module

    Arguments:
        d_hidden {int} -- dimension of hidden
        d_inner {int} -- dimension of inner hidden vector
        dropout_rate {float} -- rate for dropout
    """

    def __init__(self, d_hidden, d_inner, dropout_rate=0.1):
        super(PositionWiseFeedForward, self).__init__()

        # position-wise feed forward
        self.weight1 = nn.Conv1d(d_hidden, d_inner, 1)
        self.weight2 = nn.Conv1d(d_inner, d_hidden, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        """linear transformations are the same scross different positions,
        Another way to describe this as two convolutions with kernel size 1
        
        Args:
            x (tensor): shape (batch, len_x, d_model)
        
        Returns:
            tensor: normalized data with shape (batch, len_x, d_model)
        """
        output = self.relu(self.weight1(x.transpose(1, 2)))
        output = self.weight2(output).transpose(2, 1)
        output = self.dropout(output)
        return output

if __name__ == '__main__':
    print("Hello World")
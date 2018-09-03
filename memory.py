import torch
import torch.nn as nn
from minit import uniform_init_rnn


class AttentionGRUCell(nn.Module):
    """a attention gru cell

    Arguments:
        input_size {int} -- The number of expected features in the input 'x'
        hidden_size {int} -- The number of features in the hidden state 'h'

    Returns:
        [type] -- [description]
    """

    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W = nn.Linear(input_size, hidden_size, bias=True)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)

        uniform_init_rnn(self.Wr)
        uniform_init_rnn(self.W)
        uniform_init_rnn(self.U)
        uniform_init_rnn(self.Ur)

    def forward(self, fact, c, g) -> torch.tensor:
        """forward step of the cell

        Arguments:
            fact {number} -- shape '(batch, hidden_size)' one of the output facts input module
            c {torch.tensor} -- shape '(batch, hidden_size)'
            g {tensor} -- shape ‘(batch, )’

        Returns:
            torch.tensor -- [description]
        """
        # r shape 'batch, hidden_size'
        r = torch.sigmoid(self.Wr(fact) + self.Ur(c))
        # h_tilda shape 'batch, hidden_size'
        h_tilda = torch.tanh(self.W(fact) + r * self.U(c))
        g = g.unsqueeze(1)
        c = g * h_tilda + (1 - g) * c
        return c


class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G) -> torch.tensor:
        """forwar step

        Arguments:
            facts {tensor} -- shape '(batch, seq_len, hidden_size)'
            G {tensor} -- shape '(batch, sentence)'

        Returns:
            c {tensor} -- shape '(batch, hidden_size)'
        """
        batch_num, seq_num, hidden_size = facts.size()
        c = torch.zeros(batch_num, hidden_size, device=facts.device)
        for sid in range(seq_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            c = self.AGRUCell.forward(fact, c, g)
        return c


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()

        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)

        uniform_init_rnn(self.z1)
        uniform_init_rnn(self.z2)
        uniform_init_rnn(self.next_mem)
        # TODO: modify init way for Relu activate function

    def make_interaction(self, facts, questions, prev_memory) -> torch.tensor:
        """make interaction from different memory

        Arguments:
            facts {tenor} -- shape '(batch, seq_len hidden_size)'
            questions {tensor} -- shape '(batch, 1, hidden_size)'
            prev_memeory {tensor} -- shape '(batch, 1, hidden_size)'

        Returns:
            G {tensor} -- shape '(batch, seq_len)'
        """
        batch_num, seq_len, hidden_size = facts.size()

        # z shape 'batch, seq, 4 * hidden_size'
        z = torch.cat([facts * questions,
                       facts * prev_memory,
                       torch.abs(facts - questions),
                       torch.abs(facts - prev_memory)], dim=2)

        z = z.view(-1, 4 * hidden_size)

        G = torch.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = torch.softmax(G, dim=1)

        return G

    def forward(self, facts, questions, prev_memory) -> torch.tensor:
        """[summary]

        Arguments:
            facts {tensor} -- shape '(batch, seq, hidden_size)'
            questions {tensor} -- shape '(batch, 1, hidden_size)'
            prev_memory {tensor} -- shape '(batch, 1, hidden_size)'

        Returns:
            next_memory {tensor} -- shape '(batch, 1, hidden_size)'
        """
        G = self.make_interaction(facts, questions, prev_memory)
        C = self.AGRU.forward(facts, G)
        concat = torch.cat([prev_memory.squeeze(
            1), C, questions.squeeze(1)], dim=1)
        # print("concat size: {}".format(concat.size()))
        next_memory = self.next_mem(concat)
        next_memory = torch.relu(next_memory)
        next_memory = next_memory.unsqueeze(1)
        return next_memory

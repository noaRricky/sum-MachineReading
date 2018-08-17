import torch
import torch.nn as nn
import torch.nn.init as init

VOCAL_SIZE = 100
EMBEDED_SIZE = 30
HIDDEN_SIZE = 30


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
        init.xavier_normal_(self.Wr.weight)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=True)
        init.xavier_normal_(self.Ur.weight)
        self.W = nn.Linear(input_size, hidden_size, bias=True)
        init.xavier_normal_(self.W.weight)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        init.xavier_normal_(self.U.weight)

    def forward(self, fact, c, g) -> torch.tensor:
        """forward step of the cell

        Arguments:
            fact {number} -- shape '(batch, hidden_size)' one of the output facts input module
            c {torch.tensor} -- shape '(batch, hidden_size)'
            g {tensor} -- shape ‘(batch, )’

        Returns:
            torch.tensor -- [description]
        """
        r = torch.sigmoid(self.Wr(fact) + self.Ur(c))
        h_tilda = torch.tanh(self.W(fact) + r * self.U(c))
        g = g.unsqueeze(1).expand_as(h_tilda)
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
            facts {tensor} -- shape '(seq_len, batch, hidden_size)'
            G {tensor} -- shape '(batch, sentence)'

        Returns:
            c {tensor} -- shape '(batch, hidden_size)'
        """
        seq_num, batch_num, hidden_size = facts.size()
        c = torch.zeros(batch_num, hidden_size)
        for sid in range(seq_num):
            fact = facts[sid]
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
        init.xavier_normal_(self.z1.weight)
        init.xavier_normal_(self.z2.weight)
        init.xavier_normal_(self.next_mem.weight)
        # TODO: modify init way for Relu activate function

    def make_interaction(self, facts, questions, prev_memory) -> torch.tensor:
        """make interaction from different memory

        Arguments:
            facts {tenor} -- shape '(seq_len, batch, hidden_size)'
            questions {tensor} -- shape '(1, batch, hidden_size)'
            prev_memeory {tensor} -- shape '(1, batch, hidden_size)'

        Returns:
            G {tensor} -- shape '(batch, seq_len)'
        """
        seq_len, batch_num, hidden_size = facts.size()
        questions = questions.expand_as(facts)
        prev_memory = prev_memory.expand_as(facts)

        z = torch.cat([facts * questions,
                       facts * prev_memory,
                       torch.abs(facts - questions),
                       torch.abs(facts - prev_memory)], dim=2)

        # print("concat size: {}".format(z.size()))
        z = z.view(-1, 4 * hidden_size)
        # print("convert size: {}".format(z.size()))

        G = torch.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)

        G = torch.softmax(G, dim=1)

        return G

    def forward(self, facts, questions, prev_memory) -> torch.tensor:
        """[summary]

        Arguments:
            facts {tensor} -- shape '(seq_len, batch, hidden_size)'
            questions {tensor} -- shape '(1, batch, hidden_size)'
            prev_memory {tensor} -- shape '(1, batch, hidden_size)'

        Returns:
            next_memory {tensor} -- shape '(1, batch, hidden_size)'
        """
        G = self.make_interaction(facts, questions, prev_memory)
        C = self.AGRU.forward(facts, G)
        concat = torch.cat([prev_memory.squeeze(0), C, questions.squeeze(0)], dim=1)
        print("concat size: {}".format(concat.size()))
        next_memory = self.next_mem(concat)
        next_memory = torch.relu(next_memory)
        return next_memory


def test_attention_cell():
    attention_gru_cell = AttentionGRUCell(HIDDEN_SIZE, HIDDEN_SIZE)
    fact = torch.randn(2, HIDDEN_SIZE)
    c = torch.randn(2, HIDDEN_SIZE)
    g = torch.randn(2)
    c = attention_gru_cell.forward(fact, c, g)
    print("context size: {}".format(c.size()))


def test_attention_gru():
    input_size = HIDDEN_SIZE
    sentence_len = 4
    batch_size = 2
    attention_gru = AttentionGRU(input_size, HIDDEN_SIZE)
    facts = torch.randn(sentence_len, batch_size, input_size)
    G = torch.randn(batch_size, sentence_len)
    context = attention_gru.forward(facts, G)
    print("context size: {}".format(context.size()))
    print("context value: \n{}".format(context))


def text_episodic_memory():
    sentence_len = 4
    batch_size = 2
    episodic_memory = EpisodicMemory(HIDDEN_SIZE)
    facts = torch.randn(sentence_len, batch_size, HIDDEN_SIZE)
    questions = torch.randn(1, batch_size, HIDDEN_SIZE)
    memory = torch.randn(1, batch_size, HIDDEN_SIZE)
    g = episodic_memory.make_interaction(facts, questions, memory)
    print("g size: {}".format(g.size()))
    print("g value:\n {}".format(g))
    memory = episodic_memory.forward(facts, questions, memory)
    print("memory size: {}".format(memory.size()))


if __name__ == '__main__':
    # test_attention_gru()
    text_episodic_memory()

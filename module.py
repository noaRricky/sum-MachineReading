import torch
import torch.nn as nn

from minit import uniform_init_rnn


class QuestionModule(nn.Module):
    def __init__(self, embeded_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(embeded_size, hidden_size, batch_first=True)
        uniform_init_rnn(self.gru)

    def forward(self, questions, word_embedding) -> torch.tensor:
        """encoding the question tensor

        Arguments:
            questions {nn.Tensor} -- size: batch_num, tokens
            word_embedding {nn.Embedding} -- embedding function

        Returns:
            torch.tensor -- size: (batch, 1, hidden_size)
        """
        questions = word_embedding(questions)
        # print('embeded questions size: {}'.format(questions.size()))
        # question.size -> batch_num, seq_len, input_size
        # print('questions transpose size: {}'.format(questions.size()))
        # hidden shape: '1, batch_bum, hidden_size'
        _, hidden = self.gru(questions)
        # transpose hidden shape: 'batch, 1, hiddden_size'
        hidden = hidden.transpose(0, 1)
        return hidden


class AnswerModule(nn.Module):
    def __init__(self, vocab_size, embeded_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.gru = nn.GRU(embeded_size, hidden_size * 2, batch_first=True)
        self.z = nn.Linear(2 * hidden_size, vocab_size)

        uniform_init_rnn(self.gru)
        uniform_init_rnn(self.z)

    def forward(self, hidden, answers, embedding):
        """generate answer by memory and questions

        Arguments:
            hidden {torch.tensor} -- size: (1, batch, hidden_size * 2)
            answers {torch.tensor} -- size: (batch, 1)

        Returns:
            words {tensor} -- shape '(batch, vocab_size)'
            hidden {tensor} -- shape '(batch, 1, 2 * hidden_size)'
        """
        embeded = embedding(answers)
        output, hidden = self.gru(embeded, hidden)
        # output.size() -> batch, 1, 2 * hidden_size
        output = output.squeeze(1)
        hidden = hidden.transpose(0, 1)
        words = self.z(output)
        return words, hidden


class InputModule(nn.Module):
    def __init__(self, embeded_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embeded_size, hidden_size)

        uniform_init_rnn(self.gru)

    def forward(self, contexts, embedding: nn.Embedding) -> torch.tensor:
        """forward input contexts

        Arguments:
            contexts {torch.Tenser} -- size: (batch, token_len)
            word_embedding {nn.Embedding} -- embeddng for each idx word

        Returns:
            torch.tensor -- output.size() -> (batch, seq_len, 2 * hidden_size)
        """
        contexts = embedding(contexts)
        # contexts.size() -> (batch, seq_len, embeded_size)
        output, _ = self.gru(contexts)
        return output


if __name__ == '__main__':
    question_module = QuestionModule(124, 124)

import torch
import torch.nn as nn
import torch.nn.init as init


# TODO: implement chono init method


class QuestionModule(nn.Module):
    def __init__(self, embeded_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(embeded_size, hidden_size)

    def forward(self, questions, word_embedding) -> torch.tensor:
        """encoding the question tensor

        Arguments:
            questions {nn.Tensor} -- size: batch_num, tokens
            word_embedding {nn.Embedding} -- embedding function

        Returns:
            torch.tensor -- size: (1, batch, hidden_size)
        """
        questions = word_embedding(questions)
        # print('embeded questions size: {}'.format(questions.size()))
        # question.size -> batch_num, seq_len, input_size
        questions = questions.transpose(0, 1)
        # print('questions transpose size: {}'.format(questions.size()))
        _, questions = self.gru(questions)
        return questions


class AnswerModule(nn.Module):
    def __init__(self, vocab_size, embeded_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.gru = nn.GRU(embeded_size, hidden_size * 2)
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        init.xavier_normal_(self.z.weight)

    def forward(self, hidden, answers, embedding):
        """generate answer by memory and questions

        Arguments:
            hidden {torch.tensor} -- size: (1, batch, hidden_size * 2)
            answers {torch.tensor} -- size: (batch, 1)

        Returns:
            [type] -- [description]
        """
        embeded = embedding(answers)
        embeded = embeded.transpose(0, 1)
        output, hidden = self.gru(embeded, hidden)
        # output.size() -> 1, batch, 2 * hidden_size
        output = output.squeeze(0)
        words = self.z(output)
        return words, hidden


class InputModule(nn.Module):
    def __init__(self, embeded_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embeded_size, hidden_size)

    def forward(self, contexts, embedding: nn.Embedding) -> torch.tensor:
        """forward input contexts

        Arguments:
            contexts {torch.Tenser} -- size: (batch, token_len)
            word_embedding {nn.Embedding} -- embeddng for each idx word

        Returns:
            torch.tensor -- output.size() -> (seq_len, batch, 2 * hidden_size)
        """
        contexts = embedding(contexts)
        # contexts.size() -> (batch, seq_len, embeded_size)
        contexts = contexts.transpose(0, 1)
        output, _ = self.gru(contexts)
        return output

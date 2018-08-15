import torch
import torch.nn as nn
import torch.nn.init as init


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
        # question.size -> batch_num, seq_len, input_size
        questions = questions.transpose(0, 1)
        _, questions = self.gru(questions)
        return questions


class AnswerModule(nn.Module):
    def __init__(self, vocab_size, embeded_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.gru = nn.GRU(embeded_size, hidden_size * 2)
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        init.xavier_normal_(self.z.weight)

    def forward(self, memory, questions, answers, embedding):
        """generate answer by memory and questions

        Arguments:
            memory {torch.tensor} -- size: (1, batch, hidden_size)
            questions {torch.tensor} -- size: (1, batch, hidden_size)
            answers {torch.tensor} -- size: (batch, 1)

        Returns:
            [type] -- [description]
        """

        hidden = torch.cat([memory, questions], dim=2).squeeze(0)
        embeded = embedding(answers)
        embeded = embeded.transpose(0, 1)
        output, _ = self.gru(embeded, hidden)
        # output.size() -> 1, batch, 2 * hidden_size
        output = output.squeeze(0)
        word = self.z(output)
        return word


class InputModule(nn.Module):
    def __init__(self, embeded_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embeded_size, hidden_size, bidirectional=True)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)

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
        output, _ = self.gru(contexts)
        return output


if __name__ == '__main__':
    # input_model = InputModule(100, 30)
    VOCAL_SIZE = 100
    HIDDEN_SIZE = 30
    question_module = QuestionModule(VOCAL_SIZE, HIDDEN_SIZE)
    embedding = nn.Embedding(VOCAL_SIZE, HIDDEN_SIZE)
    questions = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 2]], dtype=torch.long)
    question_module.forward(questions, embedding)

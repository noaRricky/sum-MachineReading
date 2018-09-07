import torch
import torch.nn as nn
import torch.nn.init as init

from model.module import InputModule, QuestionModule, AnswerModule
from model.memory import EpisodicMemory
from model.constants import SOS_TOKEN, EOS_TOKEN


class DynamicMemoryNetworkPlus(nn.Module):
    """ dynamic memory network model

    Arguments:
        vocab_size {int} -- vocabulary size
        embeded_size {int} -- the embedding result dimension
        hidden_size {int} -- the hidden layer feature size
        num_hop {int} -- how many time memory module update
        device {str} -- which device use to compute tensor
    """

    def __init__(self, vocab_size, embeded_size, hidden_size, num_hop=3, qa=None):
        super(DynamicMemoryNetworkPlus, self).__init__()

        # init num file
        self.num_hop = num_hop
        self.qa = qa
        self.vocab_size = vocab_size

        # init network
        self.word_embedding = nn.Embedding(
            vocab_size, embeded_size)
        init.uniform_(self.word_embedding.weight, a=-(3 ** 0.5), b=(3 ** 0.5))
        self.criterion = nn.CrossEntropyLoss()

        self.input_module = InputModule(embeded_size, hidden_size)
        self.question_module = QuestionModule(embeded_size, hidden_size)
        self.answer_module = AnswerModule(
            vocab_size, embeded_size, hidden_size)
        self.episodic_memory = EpisodicMemory(hidden_size)

    def forward(self, contexts, questions, max_length=1000) -> torch.tensor:
        """read contexts and question to generate answer
        Arguments:
            contexts {tensor} -- shape '(batch, token)'
            questions {tensor} -- shape '(batch, token)'
            max_length {int} -- default max length of each answer

        Returns:
            hidden {tensor} --  shape '(1, batch, hidden_size)'
        """
        batch_num, _ = contexts.size()

        facts = self.input_module.forward(contexts, self.word_embedding)
        questions = self.question_module.forward(
            questions, self.word_embedding)
        memory = questions
        for hop in range(self.num_hop):
            memory = self.episodic_memory.forward(facts, questions, memory)

        # concat the memory and questions output to generate hidden vector
        hidden = torch.cat([memory, questions], dim=2)
        hidden = hidden.transpose(0, 1)

        return hidden

    def train(self, contexts, questions, max_length):
        """train operation for the network

        Arguments:
            contexts {tensor} -- shape '(batch, token)'
            questions {tensor} -- shape '(batch, token)'
            max_length {int} -- default max length of each answer

        Returns:
            preds {tensor} -- shape '(batch, seq_len, vocab_size)'
        """
        batch_num, _ = contexts.size()

        # hidden shape (1, batch, hidden_size)
        hidden = self.forward(contexts, questions, max_length)

        preds = None
        words = torch.zeros(batch_num, 1, dtype=torch.long,
                            device=contexts.device) + SOS_TOKEN
        for di in range(max_length):
            output, hidden = self.answer_module.forward(
                hidden, words, self.word_embedding)
            # get next words
            topv, topn = output.topk(1)
            words = topn.long()
            # saving output
            if di == 0:
                preds = output.unsqueeze(1)
            else:
                preds = torch.cat([preds, output.unsqueeze(1)], dim=1)
        return preds

    def predict(self, contexts, questions, max_length=1000):
        """predict operation for the network

        Arguments:
            contexts {tensor} -- shape '(batch, token)'
            questions {tensor} -- shape '(batch, token)'
            max_length {int} -- default max length of each answer

        Returns:
            preds {list} -- shape '(batch, each_answer_len)'
        """
        batch_num, _ = contexts.size()
        preds = []

        hidden = self.forward(contexts, questions, max_length)

        for bi in range(batch_num):
            word = torch.zeros(1, 1, dtype=torch.long,
                               device=contexts.device) + SOS_TOKEN
            answer = []
            each_hidden = hidden[:, bi, :].unsqueeze(0)
            for di in range(max_length):
                output, each_hidden = self.answer_module.forward(
                    each_hidden, word, self.word_embedding
                )
                topv, topn = output.topk(1)
                answer.append(topn.item())
                if topn.item() == EOS_TOKEN:
                    break
                else:
                    word = topn.long()
            preds.append(answer)
        return preds

    def loss(self, contexts, questions, answers) -> float:
        """get lost by answers

        Arguments:
            contexts {tensor} -- shape '(batch, token)'
            questions {tensor} -- shape '(batch, token)'
            answers {tensor} -- shape '(batch, token)'

        Returns:
            float -- loss value
        """
        batch_num, answers_len = answers.size()
        preds = self.train(contexts, questions, answers_len)
        preds = preds.view(batch_num * answers_len, -1)
        answers = answers.view(batch_num * answers_len)
        loss = self.criterion(preds, answers)
        return loss

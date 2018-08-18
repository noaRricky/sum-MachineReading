import torch
import torch.nn as nn
import torch.nn.init as init

from module import InputModule, QuestionModule, AnswerModule
from memory import EpisodicMemory

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 1000


class DynamicMemoryNetworkPlus(nn.Module):
    """ dynamic memory network model

    Arguments:
        vocab_size {int} -- vocabulary size
        embeded_size {int} -- the embedding result dimension
        hidden_size {int} -- the hidden layer feature size
        num_hop {int} -- how many time memory module update
    """

    def __init__(self, vocab_size, embeded_size, hidden_size, num_hop=3, qa=None):
        super(DynamicMemoryNetworkPlus, self).__init__()

        # init num file
        self.num_hop = num_hop
        self.qa = qa

        # init network
        self.word_embedding = nn.Embedding(
            vocab_size, embeded_size, padding_idx=0, sparse=True)
        init.uniform_(self.word_embedding.weight, a=-(3 ** 0.5), b=(3 ** 0.5))
        self.criterion = nn.CrossEntropyLoss()

        self.input_module = InputModule(embeded_size, hidden_size)
        self.question_module = QuestionModule(embeded_size, hidden_size)
        self.answer_module = AnswerModule(
            vocab_size, embeded_size, hidden_size)
        self.episodic_memory = EpisodicMemory(hidden_size)

    def forward(self, contexts, questions, operation, max_length=MAX_LENGTH) -> torch.tensor:
        """read contexts and question to generate answer
        Arguments:
            contexts {tensor} -- shape '(batch, token)'
            questions {tensor} -- shape '(batch, token)'
            operation {str} -- ['train', 'predict'] 
                when train the network will compute the loss
                when predict the network will generate answers
            max_length {int} -- default max length of each answer

        Returns:
            preds {tensor or list} --  if 'train' shape '(batch. seq_len, vocab_size)'
                else if 'predict' shape '(batch, token)'
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

        # print("dynamic network answer module hidden size: {}".format(hidden.size()))
        words = torch.zeros(batch_num, 1, dtype=torch.long)
        if operation == 'train':
            return self.train(words, hidden, max_length)
        elif operation == 'predict':
            return self.predict(words, hidden, max_length)

    def train(self, words, hidden, max_length):
        """train operation for the network

        Arguments:
            words {tensor} -- shape '(batch, token)'
            hidden {tensor} -- shape '(batch, hidden_size)'
            max_length {int} -- answer sequence len

        Returns:
            preds {tensor} -- shape '(batch, seq_len, vocab_size)'
        """

        preds = None
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

    def predict(self, words, hidden, max_length):
        """predict operation for the network

        Arguments:
            words {tensor} -- shape '(batch, token)'
            hidden {tensor} -- shape '(batch, token)'
            max_length {int} -- default max length for the generated answers

        Returns:
            preds {list} -- shape '(batch, each_answer_len)'
        """

        preds = []
        batch_num, _ = words.size()
        for bi in range(batch_num):
            word = words[bi].unsqueeze(0)
            answer = []
            for di in range(max_length):
                output, hidden = self.answer_module.forward(
                    hidden, word, self.word_embedding
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
        preds = self.forward(contexts, questions,
                             operation='train', max_length=answers_len)
        preds = preds.view(batch_num * answers_len, -1)
        answers = answers.view(batch_num * answers_len)
        loss = self.criterion(preds, answers)
        return loss


def train_network(data_path, dict_path):
    # TODO: train dynamic memory networkk
    print("hello world")


if __name__ == '__main__':
    print("hello world")

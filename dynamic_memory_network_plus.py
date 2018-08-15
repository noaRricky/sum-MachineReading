import torch
import torch.nn as nn
import torch.nn.init as init

from module import InputModule, QuestionModule, AnswerModule
from memory import EpisodicMemory


class DynamicMemoryNetworkPlus(nn.Module):
    def __init__(self, vocab_size, embeded_size, hidden_size):
        super(DynamicMemoryNetworkPlus, self).__init__()

        self.word_embedding = nn.Embedding(
            vocab_size, embeded_size, padding_idx=0, sparse=True)
        init.uniform_(self.word_embedding.weight, a=-(3 ** 0.5), b=(3 ** 0.5))
        self.criterion = nn.CrossEntropyLoss(size_average=False)

        self.input_module = InputModule(embeded_size, hidden_size)
        self.question_module = QuestionModule(embeded_size, hidden_size)
        self.answer_module = AnswerModule(
            vocab_size, embeded_size, hidden_size)
        self.episodic_memory = EpisodicMemory(hidden_size)

    def forward(self, contexts, questions):

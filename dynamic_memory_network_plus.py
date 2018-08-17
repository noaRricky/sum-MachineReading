import torch
import torch.nn as nn
import torch.nn.init as init

from module import InputModule, QuestionModule, AnswerModule
from memory import EpisodicMemory

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 1000


class DynamicMemoryNetworkPlus(nn.Module):
    def __init__(self, vocab_size, embeded_size, hidden_size, num_hop=3, qa=None):
        super(DynamicMemoryNetworkPlus, self).__init__()

        self.num_hop = num_hop
        self.qa = qa

        self.word_embedding = nn.Embedding(
            vocab_size, embeded_size, padding_idx=0, sparse=True)
        init.uniform_(self.word_embedding.weight, a=-(3 ** 0.5), b=(3 ** 0.5))
        self.criterion = nn.CrossEntropyLoss()

        self.input_module = InputModule(embeded_size, hidden_size)
        self.question_module = QuestionModule(embeded_size, hidden_size)
        self.answer_module = AnswerModule(
            vocab_size, embeded_size, hidden_size)
        self.episodic_memory = EpisodicMemory(hidden_size)

    def forward(self, contexts, questions) -> torch.tensor:
        """read contexts and question to generate answer

        Arguments:
            contexts {tensor} -- shape '(batch, token)'
            questions {tensor} -- shape '(batch, token)'

        Returns:
            preds {tensor} -- shape '(batch, vocab_size)'
        """
        batch_num, _ = contexts.size()

        facts = self.input_module.forward(contexts, self.word_embedding)
        questions = self.question_module.forward(
            questions, self.word_embedding)
        memory = questions
        for hop in range(self.num_hop):
            memory = self.episodic_memory.forward(facts, questions, memory)
        # o infer to SOS
        # print("memory size: {}".format(memory.size()))
        # print("quesions size: {}".format(questions.size()))
        hidden = torch.cat([memory, questions], dim=2)
        # print("dynamic network answer module hidden size: {}".format(hidden.size()))
        answers = torch.zeros(batch_num, 1, dtype=torch.long)
        preds, hidden = self.answer_module.forward(
            hidden, answers, self.word_embedding)
        return preds

    def encoder(self, context, question):
        """generate encode [memory, quesion] tensor

        Arguments:
            context {tensor} -- shape '(batch_num=1, token)'
            question {[tensor]} -- shape '(batch_num=1, token)'

        Returns:
            tensor -- shape '1, batch_num=1, hidden_size * 2'
        """

        facts = self.input_module.forward(context, self.word_embedding)
        question = self.question_module.forward(question, self.word_embedding)
        memory = question
        for hop in range(self.num_hop):
            memory = memory = self.episodic_memory.forward(
                facts, question, memory)
        hidden = torch.cat([memory, question], dim=2)
        return hidden

    def predict(self, context, question, device, max_length=MAX_LENGTH) -> torch.tensor:
        """generate answer by context and question tensor

        Arguments:
            context {tensor} -- shape '(batch=1, token)'
            question {tensor} -- shape '(batch=1, token)'
            device {str} -- device type

        Returns:
            torch.tensor -- shape '(token, )'
        """
        hidden = self.encoder(context, question)
        answer_word = torch.tensor(
            [[SOS_TOKEN]], dtype=torch.long, device=device)
        answer = []

        for di in range(MAX_LENGTH):
            output, hidden = self.answer_module.forward(
                hidden, answer_word, self.word_embedding)
            topv, topi = output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                break
            else:
                answer.append(topi.item())

    def loss(self, context, question, answer, device) -> float:
        """compute loss by the data

        Arguments:
            context {tensor} -- shape '(batch=1, token)'
            questions {tensor} -- shape '(batch=1, token)'
            answers {tensor} -- shape '(token, )'

        Returns:
            float -- the terminal loss
        """
        answer_len = answer.size()[1]
        loss = 0

        hidden = self.encoder(context, question)
        decoder_input = torch.tensor(
            [[SOS_TOKEN]], dtype=torch.long, device=device)
        for idx in range(answer_len):
            output, hidden = self.answer_module.forward(
                hidden, decoder_input, self.word_embedding)
            loss += self.criterion(output, answer[idx])
            _, decoder_input = output.topk(1)

        return loss


def train_network(data_path, dict_path):
    # TODO: train dynamic memory networkk
    print("hello world")


if __name__ == '__main__':
    print("hello world")

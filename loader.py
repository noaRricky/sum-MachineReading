import json

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from constants import QUESTION, QUESTIONS, ANSWER, ARTICLE_CONTENT, PAD_TOKEN, EOS_TOKEN, DATA_PATH


class QADataSet(Dataset):
    """reading qa dataset from data

    Arguments:
        data_path {str} -- path for saved data
    """

    def __init__(self, data_path: str):
        data = None
        with open(data_path, mode='r', encoding='utf-8') as fp:
            data = json.load(fp)

        data_gen = data_generator(data)
        dataset = [(c, q, a) for (c, q, a) in data_gen]
        contents, questions, answers = zip(*dataset)
        self.contents = list(contents)
        self.questions = list(questions)
        self.answers = list(answers)

    def __getitem__(self, index):
        """get item contains (content, question, answer)

        Arguments:
            index {int} -- postion of data
        """
        return self.contents[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.contents)


def data_generator(data):
    for article in data:
        c = article[ARTICLE_CONTENT]
        for qobj in article[QUESTIONS]:
            q = qobj[QUESTION]
            a = qobj[ANSWER]
            yield c, q, a


def pad_collate(batch):
    max_content_len = 0
    max_question_len = 0
    max_answer_len = 0
    for elem in batch:
        content, question, answer = elem
        content.append(EOS_TOKEN)
        question.append(EOS_TOKEN)
        answer.append(EOS_TOKEN)
        max_content_len = max_content_len if max_content_len > len(
            content) else len(content)
        max_question_len = max_content_len if max_question_len > len(
            question) else len(question)
        max_answer_len = max_answer_len if max_answer_len > len(
            answer) else len(answer)

    for i, elem in enumerate(batch):
        content, question, answer = elem
        content = np.pad(content, (0, max_content_len - len(content)),
                         mode='constant', constant_values=PAD_TOKEN)
        question = np.pad(question, (0, max_question_len - len(question)),
                          mode='constant', constant_values=PAD_TOKEN)
        answer = np.pad(answer, (0, max_answer_len - len(answer)),
                        mode='constant', constant_values=PAD_TOKEN)
        batch[i] = (content, question, answer)
    return default_collate(batch)


if __name__ == '__main__':
    dataset = QADataSet(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=2,
                            shuffle=True, collate_fn=pad_collate)
    for c, q, a in dataloader:
        print(a)
        break

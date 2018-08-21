import json

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import constants


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
        c = article[constants.ARTICLE_CONTENT]
        for qobj in article[constants.QUESTIONS]:
            q = qobj[constants.QUESTION]
            a = qobj[constants.ANSWER]
            yield c, q, a


def pad_collate(batch):
    for elem in batch:
        print("element: {}".format(element))
    return default_collate()

if __name__ == '__main__':
    dataset = QADataSet(constants.DATA_PATH)
    dataloader = DataLoader(dataset, shuffle=True)
    print(len(dataset))

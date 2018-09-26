import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from gensim.corpora import Dictionary

from model import DynamicMemoryNetworkPlus
from loader import QADataSet, pad_collate

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# setting for file
DATA_PATH = './data/data_idx.json'
DIC_PATH = './data/jieba.dict'


def train_network():
    # get vocab size
    dictionary: Dictionary = Dictionary.load(DIC_PATH)
    vocab_size = len(dictionary.token2id) + 4
    del dictionary

    # select device to train
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info("use {} to train network".format(device))

    # hyperparameter for network
    embeding_size = 256
    hidden_size = 256
    learning_rate = 0.003
    num_epoch = 256

    # initlise the network
    logging.info("init the dynamic memory model")
    model = DynamicMemoryNetworkPlus(vocab_size, embeding_size, hidden_size)
    model.to(device)

    # seting the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_total = 0

    # load dataset
    logging.info("loading train dataset")
    dataset = QADataSet(DATA_PATH)
    print("dataset size: {}".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, collate_fn=pad_collate)

    for iter_idx in range(num_epoch):
        for idx, (content, question, answer) in enumerate(dataloader):
            # zero the parameter grad
            optimizer.zero_grad()

            # feed to trainig device
            content = content.long().to(device)
            question = question.long().to(device)
            answer = answer.long().to(device)

            # forward \
            loss = model.loss(content, question, answer)

            # backword and optimize
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

            # print loss
            if idx % 10 == 9:  # print every 10 mini-batch
                logging.info("epoch {}, item {}, loss {}".format(
                    iter_idx, idx, loss_total / 10))
                loss_total = 0

    return model


if __name__ == '__main__':
    model = train_network()
    torch.save(model, './data/dmnp.mdl')

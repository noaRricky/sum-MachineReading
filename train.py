import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from gensim.corpora import Dictionary

from dynamic_memory_network_plus import DynamicMemoryNetworkPlus
from loader import QADataSet, pad_collate
from constants import DATA_PATH, DIC_PATH, EXTRA_SIZE

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_network():
    # get vocab size
    dictionary: Dictionary = Dictionary.load(DIC_PATH)
    vocab_size = len(dictionary.token2id) + EXTRA_SIZE
    del dictionary

    # hyperparameter for network
    embeding_size = 256
    hidden_size = 256
    learning_rate = 0.003
    num_epoch = 256

    # initlise the network
    logging.info("init the dynamic memory model")
    model = DynamicMemoryNetworkPlus(vocab_size, embeding_size, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_total = 0

    # load dataset
    logging.info("loading train dataset")
    dataset = QADataSet(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, collate_fn=pad_collate)

    for iter_idx in range(num_epoch):
        for idx, (content, question, answer) in enumerate(dataloader):
            # zero the parameter grad
            optimizer.zero_grad()

            # forward \
            loss = model.loss(content, question, answer)

            # backword and optimize
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

            # print loss
            if idx % 100 == 99:  # print every 10 mini-batch
                logging.info("epoch {}, item {}, loss {}".format(
                    iter_idx, idx, loss_total / 100))
                loss_total = 0

    return model


if __name__ == '__main__':
    model = train_network()
    torch.save(model, './data/dmnp.mdl')

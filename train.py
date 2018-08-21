import logging

import torch
import torch.optim as optim

from dynamic_memory_network_plus import DynamicMemoryNetworkPlus

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# setting for file
DATA_PATH = './data/data_idx.json'


def train_network():

    vocab_size = len(dictionary.token2id)
    # train the network
    model = DynamicMemoryNetworkPlus(vocab_size, EMBEDDED_SIZE, HIDDEN_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_total = 0

    for iter_idx in range(NUM_EPOCH):
        item_gen = get_item(data)
        for idx, (content, question, answer) in enumerate(item_gen):
            # zero the parameter grad
            optimizer.zero_grad()

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

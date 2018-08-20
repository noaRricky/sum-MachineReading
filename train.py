import json
import logging

from gensim.corpora import Dictionary
import torch
import torch.optim as optim

from dynamic_memory_network_plus import DynamicMemoryNetworkPlus

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# training setting
EMBEDDED_SIZE = 124
HIDDEN_SIZE = 256
NUM_EPOCH = 100
MAX_LENGTH = 1000
LEARNING_RATE = 0.003

# setting for data
ARTICLE_ID = 'article_id'
ARTICLE_TITLE = 'article_title'
ARTICLE_CONTENT = 'article_content'
QUESTIONS = 'questions'
QUESTION = 'question'
QUESTIONS_ID = 'questions_id'
ANSWER = 'answer'

# setting for file
DATA_PATH = './data/data_idx.json'
DIC_PATH = './data/jieba.dict'


def get_item(data):
    for article in data:
        content = torch.tensor([article[ARTICLE_CONTENT]], dtype=torch.long)
        for qobj in article[QUESTIONS]:
            question = torch.tensor([qobj[QUESTION]], dtype=torch.long)
            answer = torch.tensor([qobj[ANSWER]], dtype=torch.long)
            yield content, question, answer


def print_data(data):
    """check data length

    Arguments:
        data {dict} -- data
    """

    for article in data:
        if len(article[ARTICLE_CONTENT]) == 1:
            print("article:\n{}".format(article[ARTICLE_CONTENT]))
        for qobj in article[QUESTIONS]:
            if len(qobj[QUESTION]) == 1 or len(qobj[ANSWER]) == 1:
                print("question:\n{}".format(qobj[QUESTION]))
                print("answer:\n{}".format(qobj[ANSWER]))


def train_network(data_path, dict_path):
    # load dictionary and data
    dictionary: Dictionary = Dictionary.load(dict_path)
    with open(data_path, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)

    vocab_size = len(dictionary.token2id)
    # train the network
    model = DynamicMemoryNetworkPlus(vocab_size, EMBEDDED_SIZE, HIDDEN_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

            # print loss
            if idx % 10 == 0:  # print every 10 mini-batch
                logging.info("epoch {}, item {}, loss {}".format(
                    iter_idx, idx, loss.item()))

    return model


if __name__ == '__main__':
    model = train_network(DATA_PATH, DIC_PATH)
    torch.save(model, './data/dmnp.mdl')
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
HIDDEN_SIZE = 124
NUM_ITER = 4
MAX_LENGTH = 1000
LEARNING_RATE = 0.01

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


def get_question_len(data):
    question_len = 0
    for article in data:
        for qobj in article[QUESTIONS]:
            question_len += 1
    return question_len


def train_network(data_path, dict_path):
    # load dictionary and data
    dictionary: Dictionary = Dictionary.load(dict_path)
    with open(data_path, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)

    vocab_size = len(dictionary.token2id)
    # train the network
    model = DynamicMemoryNetworkPlus(vocab_size, EMBEDDED_SIZE, HIDDEN_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for iter_idx in range(NUM_ITER):
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

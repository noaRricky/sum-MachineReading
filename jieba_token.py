import json
import logging

import jieba
from gensim.corpora import Dictionary

ARTICLE_ID = 'article_id'
ARTICLE_TITLE = 'article_title'
ARTICLE_CONTENT = 'article_content'
QUESTIONS = 'questions'
QUESTION = 'question'
QUESTIONS_ID = 'questions_id'
ANSWER = 'answer'

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def jieba_tokenize(text: str) -> list:
    """use jieba module to tokenize word from raw text

    Arguments:
        text {str} -- raw text
    """

    text = str_q2b(text)
    tokens = list(jieba.cut(text, cut_all=False))
    tokens = [w for w in tokens if w != ' ']
    return tokens


def text_gen(file_path: str) -> str:
    """extract raw text from file

    Arguments:
        file_path {str} -- file path for saving data

    Returns:
        str -- convert from full to half
    """
    data = None
    with open(file_path, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)

    for article in data:
        # extract article text
        article_title = article[ARTICLE_TITLE]
        article_content = article[ARTICLE_CONTENT]

        yield article_title
        yield article_content

        for questions_obj in article[QUESTIONS]:
            # extract question text
            question = questions_obj[QUESTION]
            answer = questions_obj[ANSWER]

            yield question
            yield answer


def build_corpus(file_path: str, dictionary: Dictionary) -> list:
    """build data corpus convert each word to id

    Arguments:
        file_path {str} -- data file path
        dictionary {Dictionary} -- build up word2id dictionary

    Returns:
        list -- idx data list
    """

    with open(file_path, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)

    data_size = len(data)
    logging.info("data have {} articles".format(data_size))

    for idx, article in enumerate(data):
        article_content = article[ARTICLE_CONTENT]
        article_content = jieba_tokenize(article_content)
        article_content = dictionary.doc2idx(article_content)
        article_title = article[ARTICLE_TITLE]
        article_title = jieba_tokenize(article_title)
        article_title = dictionary.doc2idx(article_title)
        article[ARTICLE_CONTENT] = article_content
        article[ARTICLE_TITLE] = article_title

        for questions_obj in article[QUESTIONS]:
            question = questions_obj[QUESTION]
            question = jieba_tokenize(question)
            question = dictionary.doc2idx(question)
            questions_obj[QUESTION] = question
            answer = questions_obj[ANSWER]
            answer = jieba_tokenize(answer)
            answer = dictionary.doc2idx(answer)
            questions_obj[ANSWER] = answer

        if idx % 100 == 0:
            percent = idx / data_size
            logging.info("finish {}% of data".format(percent * 100))

    return data


def build_dictionary(file_path: str) -> Dictionary:
    """build token2id dictionary by the data in file_path

    Arguments:
        file_path {str} -- path for data file
    """
    # init dictionary
    dictionary = Dictionary()

    # build raw text generator
    text_generator = text_gen(file_path)
    for text in text_generator():
        segment = jieba_tokenize(text)
        dictionary.add_documents([segment])


def str_q2b(ustring: str) -> str:
    """convert full string to half string

    Arguments:
        string {str} -- unicode string
    """
    ret_str = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248

        ret_str += chr(inside_code)
    return ret_str


if __name__ == '__main__':
    # dictionary = build_dictionary('./data/question.json')
    # dictionary.save('./data/jieba.dict')
    dictionary = Dictionary.load('./data/jieba.dict')
    data = build_corpus('./data/question.json', dictionary)
    with open('./data/data.json', mode='w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False)

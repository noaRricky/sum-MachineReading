import json
import logging

import jieba
from gensim.corpora import Dictionary
from gensim.models import Word2Vec

# token id
SOS_TOKEN = 0
EOS_TOKEN = 1
UNKNOW = -1

# setting for data
ARTICLE_ID = 'article_id'
ARTICLE_TITLE = 'article_title'
ARTICLE_CONTENT = 'article_content'
QUESTIONS = 'questions'
QUESTION = 'question'
QUESTIONS_ID = 'questions_id'
ANSWER = 'answer'

# setting for embedding
WINDOW_SIZE = 6
EMBEDDING_SIZE = 124

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def jieba_tokenize(text: str) -> list:
    """use jieba module to tokenize word from raw text

    Arguments:
        text {str} -- raw text
    """

    text = str_q2b(text)  # convert full str into half str
    text = text.lower()  # lowercase
    tokens = list(jieba.cut(text, cut_all=False))  # generate segment by jieba
    tokens = [w for w in tokens if w != ' ']  # fliter ' ' str
    return tokens


def build_str_corpus(file_path: str) -> list:
    """build data corpus convert each word to token str

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
        article_title = article[ARTICLE_TITLE]
        article_title = jieba_tokenize(article_title)
        article[ARTICLE_CONTENT] = article_content
        article[ARTICLE_TITLE] = article_title

        for questions_obj in article[QUESTIONS]:
            question = questions_obj[QUESTION]
            question = jieba_tokenize(question)
            questions_obj[QUESTION] = question
            answer = questions_obj[ANSWER]
            answer = jieba_tokenize(answer)
            questions_obj[ANSWER] = answer

        if idx % 100 == 0:
            percent = idx / data_size
            logging.info("finish {}% of data".format(percent * 100))

    return data


def build_token_corpus(file_path: str, dictionary: Dictionary) -> list:
    """build data corpus convert each word to id and append EOS

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
        content_token = dictionary.doc2idx(article_content)
        content_token.append(EOS_TOKEN)
        article_title = article[ARTICLE_TITLE]
        article_title = jieba_tokenize(article_title)
        title_token = dictionary.doc2idx(article_title)
        title_token.append(EOS_TOKEN)
        article[ARTICLE_CONTENT] = content_token
        article[ARTICLE_TITLE] = title_token

        for questions_obj in article[QUESTIONS]:
            question = questions_obj[QUESTION]
            question = jieba_tokenize(question)
            question_token = dictionary.doc2idx(question)
            question_token.append(EOS_TOKEN)
            questions_obj[QUESTION] = question_token
            answer = questions_obj[ANSWER]
            answer = jieba_tokenize(answer)
            answer_token = dictionary.doc2idx(answer)
            answer_token.append(EOS_TOKEN)
            questions_obj[ANSWER] = answer_token

        if idx % 100 == 0:
            percent = idx / data_size
            logging.info("finish {}% of data".format(percent * 100))

    return data


def build_dictionary(file_path: str) -> Dictionary:
    """build token2id dictionary by the data in file_path

    Arguments:
        file_path {str} -- path for data file
    """
    # build raw text generator
    text_generator = text_gen(file_path)
    dictionary = Dictionary([jieba_tokenize(sentence)
                             for sentence in text_generator])
    for key in dictionary.token2id:
        dictionary.token2id[key] += 2
    return dictionary


def build_word2vec(file_path: str, dictionary: Dictionary, size: int, window=5, min_count=5, workers=4) -> Word2Vec:
    """bulld word2vec model

    Arguments:
        file_path {str} -- data file path
        size {int} -- vocabulary size

    Keyword Arguments:
        window {int} -- window size (default: {5})
        min_count {int} -- min count number (default: {1})
        worker {int} -- process to use (default: {4})

    Returns:
        Word2Vec -- word2vec model
    """

    # init word2vec model

    gen = text_gen(file_path)
    sentences = [sent for sent in gen]
    model = Word2Vec(sentences, size=size, window=window,
                     min_count=min_count, workers=workers)

    return model


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
    # data = build_corpus('./data/question.json')
    data = build_token_corpus('./data/question.json', dictionary)
    with open('./data/data_idx.json', mode='w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False)
    # file_path = './data/data.json'
    # save_path = './data/word2vec.kv'
    # model = build_word2vec(file_path, EMBEDDING_SIZE, WINDOW_SIZE)
    # model.wv.save(save_path)

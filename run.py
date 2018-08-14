import json

import pynlpir
from gensim.corpora import Dictionary

ARTICLE_ID = 'article_id'
ARTICLE_TITLE = 'article_title'
ARTICLE_CONTENT = 'article_content'
QUESTIONS = 'questions'
QUESTION = 'question'
QUESTIONS_ID = 'questions_id'
ANSWER = 'answer'


def build_dictionary(file_path: str):
    """build token2id dictionary by the data in file_path

    Arguments:
        file_path {str} -- path for data file
    """
    data = None
    with open(file_path, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)

    # build dictionary
    dictionary = Dictionary()
    with pynlpir.open():
        for article in data:
            # extract article text
            article_title = article[ARTICLE_TITLE]
            article_content = article[ARTICLE_CONTENT]

            # convert full str to half str
            article_title = str_q2b(article_title)
            article_content = str_q2b(article_content)

            # segement text
            segment1 = pynlpir.segment(article_title, pos_tagging=False)
            segment2 = pynlpir.segment(article_content, pos_tagging=False)

            # remove empty item
            segment1 = segment1.remove(' ')
            segment2 = segment2.remove(' ')

            dictionary.add_documents([segment1, segment2])

            for questions_obj in article[QUESTIONS]:
                # extract question text
                question = questions_obj[QUESTION]
                answer = questions_obj[ANSWER]

                # convert full str to half string
                question = str_q2b(question)
                answer = str_q2b(answer)

                # get token
                segment1 = pynlpir.segment(question, pos_tagging=False)
                segment2 = pynlpir.segment(answer, pos_tagging=False)

                segment1 = segment1.remove(' ')
                segment2 = segment2.remove(' ')
                dictionary.add_documents([segment1, segment2])


def str_q2b(ustring: str):
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
    text_str = "ａｂｃ－＋－-ＡＢＣ１２３你好世界，。！、"
    print(str_q2b(text_str))

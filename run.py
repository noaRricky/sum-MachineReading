import json

from pynlpir_tokenizer import Tokenzier

ARTICLE_ID = 'article_id'
ARTICLE_TITLE = 'article_title'
ARTICLE_CONTENT = 'article_content'
QUESTIONS = 'questions'
QUESTION = 'question'
QUESTIONS_ID = 'questions_id'
ANSWER = 'answer'


def extract_text(data: dict):
    """extract text from data

    Arguments:
        data {dict} -- dictionary type for saving data
    """
    for article in data:
        article_id = article[ARTICLE_ID]
        article_title = article[ARTICLE_TITLE]
        article_content = article[ARTICLE_CONTENT]
        for questions_obj in article[QUESTIONS]:
            question = questions_obj[QUESTION]
            answer = questions_obj[ANSWER] 
            


if __name__ == '__main__':
    with open("../data/question.json", mode='r', encoding='utf-8') as data_fp:
        data = json.load(data_fp)

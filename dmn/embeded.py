'''
structure of question.json:
--article_id
--article_type
--article_title
--article_content
--questions
	--questions_id
	--question
	--answer
	--question_type
'''

import json
import os
import jieba
from gensim.corpora import Dictionary
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence


class Question:
	def __init__(self,jsonstr):
		self.questions_id=jsonstr['questions_id']
		self.question=jsonstr['question']
		self.answer=jsonstr['answer']
		self.question_typoe=jsonstr['question_type']

class Article:
	def __init__(self,jsonstr):
		self.article_id=jsonstr['article_id']
		self.article_type=jsonstr['article_type']
		self.article_title=jsonstr['article_title']
		self.article_content=jsonstr['article_content']
		self.qslist=[]
		question_jsonarray=jsonstr['questions']
		for ques in question_jsonarray:
			qs=Question(ques)
			self.qslist.append(qs)

def readData():
	data=None
	with open('./data/question.json','r') as f:
		data=json.load(f)
	return data

def calType():
	article_type=[]
	question_type=[]
	data=readData()
	for article in data:
		#print(article)
		art_type=article['article_type']
		#print(art_type)
		if art_type not in article_type:
			article_type.append(art_type)
		question_jsonarray=article['questions']
		#print(question_jsonarray[0])
		#question_json=json.loads(question_jsonarray[0])
		for ques in question_jsonarray:
			ques_type=ques['question_type']
			if ques_type not in question_type:
				question_type.append(ques_type)
	print("article_type:",len(article_type))
	print("question_type:",len(question_type))

def processdata():
	data=readData()
	aslist=[]
	for article in data:
		ati=Article(article)
		aslist.append(ati)
	#print(len(aslist))
	for i in range(20000):
		seg_list=jieba.cut(aslist[i].article_content,cut_all=True)
		print("article: ",i," ".join(seg_list))


def saveToFile(filename,str):
	fo=open(filename,'a')
	fo.write(str)


def build_corpus():
	data=readData()
	aslist=[]
	for article in data:
		ati=Article(article)
		aslist.append(ati)
	#print(len(aslist))
	for i in range(20000):
		seg_list=jieba.cut(aslist[i].article_content,cut_all=True)
		str=" ".join(seg_list)
		saveToFile('./data/corpus1.txt',str)

def trainVec():
	sentences = LineSentence('./data/corpus1.txt')
	model = Word2Vec(sentences, size=300, window=8, min_count=10, sg=1, workers=4)  # sg=0 使用cbow训练, sg=1对低频词较为敏感
	model.save('./data/vec1.300d.txt')

if __name__=='__main__':
	build_corpus()
	trainVec()
	#build_corpus()
	#testjieba()
	#calType()
	#print(os.getcwd())
	#data=readData()
	#print(len(data))

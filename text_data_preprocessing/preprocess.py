#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re
import pandas as pd
import nltk

from nltk.corpus import stopwords

stopwords_dic = set(stopwords.words('english')) | {'.', ':', '(', ')', '\n'}


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST
	Original taken from https://github.com/yoonkim/CNN_Sentence/blob/master/process_data.py
	:param string:
	:return:
	"""
	# string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"[^a-zA-Z0-9.]", " ", string)
	string = re.sub(r"\'s", " is", string)
	string = re.sub(r"\'ve", " have", string)
	string = re.sub(r"n\'t", "n not", string)
	string = re.sub(r"\'re", " are", string)
	string = re.sub(r"\'d", " would", string)
	string = re.sub(r"\'ll", " will", string)
	string = re.sub(r"\'m", " am", string)
	# string = re.sub(r",", " , ", string)
	# string = re.sub(r"!", " ! ", string)
	# string = re.sub(r"\.", " . ", string)
	# string = re.sub(r"\(", " ( ", string)
	# string = re.sub(r"\)", " ) ", string)
	# string = re.sub(r"\?", " ? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"([A-Za-z0-9][a-z])([A-Z])", lambda x: x.group(1) + " " + x.group(2), string)
	string = re.sub(r"([A-Za-z0-9][a-z])([A-Z])", lambda x: x.group(1) + " " + x.group(2), string)
	# string = re.sub(r"([A-Za-z0-9]+[a-z])([A-Z][a-z])", lambda x: x.group(1) + " " + x.group(2), string)
	return string.strip().lower()


def tokenize_and_stopwords(text='', stopworddic=None, pattern=None):
	'''
	对文本进行分词并去除停用词，使用nltk工具
	:param pattern: 分词的正则表达式，有默认值
	:param stopworddic: 提供的stopwords库，默认使用NLTK中的停用词库
	:param text: 待分词的文本，一个字符串
	:return: 分词后的结果，一个数组
	'''
	if stopworddic is None:
		stopworddic = stopwords_dic
	# print(stopworddic)
	# stopworddic.add({'.', ':', '(', ')'})
	if pattern is None:
		pattern = r"""(?x)
							(?:[A-Z]\.)+
							|\d+(?:\.\d+)+%?
							|\w+(?:[-']\w+)*
							|\.\.\.
							|(?:[.,;"'?():-_`])	
						"""
	text = clean_str(text)
	result = nltk.regexp_tokenize(text, pattern)
	result = [i for i in result if i not in stopworddic]

	# result = np.asarray(result, dtype=str)

	return result


def preprocess(data):
	'''
	pre-process loaded data with tool NLTK.
	including : 分词、去除停用词、词性还原；
	:param data: 拼接了summary和description的数据
	:return:
	'''
	data['summary'] = data['summary'].map(lambda x: x[x.find(' ', 4):])
	data['desc'] = data.apply(lambda x: x['summary'] + ' ' + x['description'], axis=1)

	data.drop(labels=['summary', 'description'], axis=1, inplace=True)

	# 预处理
	data['desc'] = data['desc'].map(lambda x: tokenize_and_stopwords(x))

	return data


def load_data_from_xsl(file_name):
	"""
	Loads bug reports data from xsl files
	:param file_name:
	:return:
	"""
	nlp_data = pd.read_excel(file_name, sheet_name=0, header=0, usecols=[1, 2, 3],
	                         converters={'bug_id': str, 'summary': str, 'description': str})
	nlp_data.fillna(' ', inplace=True)

	# nlp_data['description'] = nlp_data['description'].map(lambda x: clean_str(x+''))

	return nlp_data


def save_data_to_xsl(file_name, data):
	'''
	将处理好的数据存到Excel中，即获得的vector
	:param file_name:
	:param data:
	:return:
	'''
	data.to_excel(file_name, index=False, header=True, columns=['bug_id', 'desc'])


if __name__ == '__main__':
	input_files = ['AspectJ', 'Eclipse_Platform_UI', 'JDT', 'SWT', 'Tomcat']
	main_dir = '../datasets/'

	for file in input_files:
		input_file = main_dir + file + "/" + file + '.xlsx'
		output_file = main_dir + file + "/" + file + '_pre.json'
		nlp_data = load_data_from_xsl(input_file)
		nlp_data = preprocess(nlp_data)
		nlp_data.to_json(output_file, orient='records')
		print("preprocess end for" + "\t" + file)


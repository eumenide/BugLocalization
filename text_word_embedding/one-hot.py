#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
from keras.preprocessing.text import Tokenizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


def get_one_hot_vector(data, nums_words=None):
	"""
	generate one-hot vector for data
	:param data: the array of data should be encoded
	:param nums_words: the size of vector
	:return: the matrix of one-hot vectors
	"""

	tokenizer = Tokenizer(num_words=nums_words)
	tokenizer.fit_on_texts(data)
	# sequences = tokenizer.texts_to_sequences(data)
	one_hot_vectors = tokenizer.texts_to_matrix(data, mode='binary')

	return one_hot_vectors, tokenizer.word_index


if __name__ == '__main__':
	print("main")
	sentences = ["Human machine interface for lab's abc computer applications",
	             "A survey of user opinion of computer system response time",
	             "The EPS user interface management system",
	             "System and human system engineering testing of EPS",
	             "Relation of user perceived response time to error measurement",
	             "The generation of random binary unordered trees",
	             "The intersection graph of paths in trees",
	             "Graph minors IV Widths of trees and well quasi ordering",
	             "Graph minors A survey"]

	one_hot_vector, word_index = get_one_hot_vector(sentences)

	print(word_index)

	# # 创建一个分词器（tokenizer），可设置只考虑前n个最常见的单词
	# tokenizer = Tokenizer()
	# # 构建索引单词
	# tokenizer.fit_on_texts(sentences)
	# # 将字符串转换为整数索引组成的列表
	# sequences = tokenizer.texts_to_sequences(sentences)
	# print(sequences)
	# # 可以直接得到one-hot二进制表示。这个分词器也支持除one-hot编码外其他向量化模式
	# one_hot_results = tokenizer.texts_to_matrix(sentences, mode='binary')
	# # 得到单词索引
	# word_index = tokenizer.word_index
	# print(one_hot_results)

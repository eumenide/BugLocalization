#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import json
from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format('../models/enwiki_20180420_100d.txt.bz2', binary=False)

print('load word2vec model successfully')

with open('../models/add_vocab.json', 'r') as load_f:
	add_vocab = json.load(load_f)


def my_embedding(data):
	temp = 0
	vec = []
	for string in data:
		if string in model.vocab:
			vec.append(list(model[string]))
		elif string in add_vocab:
			vec.append(list(add_vocab[string]))
		else:
			ran = (list(np.random.uniform(-1, 1, 100)))
			# ran = list(np.full([100], temp, float))
			vec.append(ran)
			add_vocab[string] = ran
			print(str(temp) + "\t" + string)

	return vec


def my_word2vec(data):
	'''
	对传入数据的desc属性进行词嵌入操作
	:param data:
	:return:
	'''
	data = data.map(lambda x : my_embedding(x))

	with open('../models/add_vocab.json', 'w') as dump_f:
		json.dump(add_vocab, dump_f)

	return data


if __name__ == '__main__':
	# input_files = ['Tomcat']
	input_files = ['AspectJ', 'Eclipse_Platform_UI', 'JDT', 'SWT', 'Tomcat']

	main_dir = '../datasets/'

	print('start embedding in directory \t' + main_dir)

	for file in input_files:
		input_file = main_dir + file + "/" + file + "_pre.json"
		output_file = main_dir + file + "/" + file + "_vec.json"
		nlp_data = pd.read_json(input_file, orient='frame', dtype=False)
		nlp_data['desc'] = my_word2vec(nlp_data['desc'])
		nlp_data.to_json(output_file, orient='records')
		print('embedding end for ' + "\t" + file)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
import logging
import traceback

import numpy as np
import threading

from gensim.models import KeyedVectors

from code_data_preprocessing.tf_idf import get_logger

DEBUG = False

if not DEBUG:
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	total_files = {'aspectj': 2394, 'eclipseUI': 17809, 'jdt': 16302, 'swt': 8560, 'tomcat': 2567}

	input_root_dir = '../datasets/SourceFile_pre/'
	projects = ['aspectj', 'eclipseUI', 'jdt', 'swt', 'tomcat']
	output_root_dir = '../datasets/'
	output_root_files = {'aspectj': 'AspectJ', 'eclipseUI': 'Eclipse_Platform_UI', 'jdt': 'JDT', 'swt': 'SWT', 'tomcat': 'Tomcat'}

	model = KeyedVectors.load_word2vec_format('../models/enwiki_20180420_100d.txt.bz2', binary=False)

	logging.info('load word2vec model successfully')

	with open('../models/add_vocab.json', 'r') as load_f:
		add_vocab = json.load(load_f)
		logging.info('load add_vocab successfully')


def word2vec_thread(project_file, logger):
	'''

	:param project_file:
	:param logger:
	:return:
	'''

	with open(input_root_dir + project_file + '.txt', 'r') as f:
		input_files = f.readlines()
		logger.info('open ' + project_file + ' file list successfully')

	with open(input_root_dir + project_file + '_w.json', 'r') as f:
		weight_list = json.load(f)

	index = 0
	project_vec = []
	add_project_vocab = {}
	for file in input_files:
		weight_dict = weight_list[index]
		file = file.strip('\n')

		file_vec = []

		with open(input_root_dir + project_file + '/' + file, 'r') as f:
			try:
				input_lines = f.readlines()
			except Exception:
				logger.error(project_file + "\n" + traceback.print_exc())

		for input_line in input_lines:
			word_list = input_line.strip('\n').split(' ')
			line_len = len(word_list)

			line_vec = np.zeros(shape=[100])

			for word in word_list:
				if word in model.vocab:
					line_vec += np.array(model[word]) * weight_dict[word]
				elif word in add_vocab:
					line_vec += np.array(add_vocab[word]) * weight_dict[word]
				elif word in add_project_vocab:
					line_vec += np.array(add_project_vocab[word]) * weight_dict[word]
				else:
					ran = np.random.uniform(-1, 1, 100)
					line_vec += ran * weight_dict[word]
					add_project_vocab[word] = list(ran)

			line_vec /= line_len
			file_vec.append(list(line_vec))
		project_vec.append({'file_id': index, 'file_vec': file_vec})
		index += 1

		logger.info(project_file + "    " + str(index) + " / " + str(total_files[project_file]))

	vector_file = output_root_dir + output_root_files[project_file] + '/' + project_file + '_code_vec.json'
	with open(vector_file, 'w') as f:
		json.dump(project_vec, f)

	add_project_file = output_root_dir + output_root_files[project_file] + '/' + project_file + '_add_vocab.json'
	with open(add_project_file, 'w') as f:
		json.dump(add_project_vocab, f)

	logger.info(project_file + "   word embedding end")


class WordThread(threading.Thread):

	def __init__(self, project_file, logger):
		threading.Thread.__init__(self)
		self.project_file = project_file
		self.logger = logger

	def run(self):
		self.logger.info('Word Embedding for code')
		word2vec_thread(self.project_file, self.logger)


if __name__ == '__main__':
	logger = get_logger('code_word_embedding', '../datasets/code_embedding_seq_3.log')

	# projects = ['swt']
	#
	# for project in projects:
	# 	word2vec_thread(project, logger)

	# multithread will break down on my computer
		# because the memory of my computer is not enough
	for file in projects:
		thread = WordThread(file, logger)
		thread.setName(file)
		thread.start()

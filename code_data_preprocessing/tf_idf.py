#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import math
import re
import threading
from collections import Counter

import os

from text_data_preprocessing import preprocess
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


total_files = {'aspectj': 1406, 'eclipseUI': 15179, 'jdt': 12682, 'swt': 8119, 'tomcat': 2355}

input_root_dir = '../datasets/SourceFile_trim/'
files = ['aspectj', 'eclipseUI', 'jdt', 'swt', 'tomcat']
output_root_dir = '../datasets/SourceFile_pre/'


def generate_tf(input_file, output_file):
	'''
	generate the tf of the terms in document
	:param input_file:
	:param output_file:
	:return:
	'''
	dirs = os.path.dirname(output_file)
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	counters = Counter()

	with open(input_file, 'r') as f:
		lines = f.readlines()
		with open(output_file, 'w') as fout:
			for line in lines:
				line = line.strip()
				line = preprocess.tokenize_and_stopwords(line)
				if not len(line):
					continue
				fout.write(' '.join(list(line)))
				fout.write('\n')
				counters += Counter(line)

	return counters


def generate_idf(word, tf_list):
	'''

	:param word:
	:param tf_list:
	:return:
	'''
	n_doc = sum(1 for tf in tf_list if word in tf)

	return math.log(len(tf_list) / n_doc + 0.01)


class TFThread(threading.Thread):

	def __init__(self, project_file, logger):
		threading.Thread.__init__(self)
		self.project_file = project_file
		self.logger = logger

	def run(self):
		self.logger.info('Calculating tf')
		tf_thread(self.project_file, self.logger)


class TFIDFThread(threading.Thread):

	def __init__(self, project_file, logger):
		threading.Thread.__init__(self)
		self.project_file = project_file
		self.logger = logger

	def run(self):
		self.logger.info('Calculating tf-idf')
		weight_thread(self.project_file, self.logger)


def get_logger(logger_name, log_file):
	'''

	:param logger_name:
	:param log_file:
	:return:
	'''
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.INFO)

	fh = logging.FileHandler(log_file)
	fmt = '%(asctime)s : %(threadName)s : %(levelname)s : %(message)s'
	formatter = logging.Formatter(fmt)
	fh.setFormatter(formatter)

	logger.addHandler(fh)
	return logger


def tf_thread(project_file, logger):
	'''

	:param project_file:
	:param logger:
	:return:
	'''
	counts = []
	temp = 1
	with open(input_root_dir + project_file + '.txt', 'r') as f:
		input_files = f.readlines()
		with open(output_root_dir + project_file + '.txt', 'a') as f2:
			for input_file in input_files:
				input_file = input_file.strip('\n')
				output_file = re.sub(r'(\.java)', '.txt', input_file)
				f2.write(output_file + '\n')
				input_file = input_root_dir + project_file + '/' + input_file
				output_file = output_root_dir + project_file + '/' + output_file
				counts.append(generate_tf(input_file, output_file))

				logger.info(project_file + "    " + str(temp) + " / " + str(total_files[project_file]))
				temp += 1

	tf_file = output_root_dir + project_file + '_tf.json'
	with open(tf_file, 'w') as f:
		counts = json.dumps(counts)
		f.write(counts)

	print('project ' + project_file + ' end')


def weight_thread(project_file, logger=None):
	'''

	:param project_file:
	:param logger:
	:return:
	'''
	temp = 1
	idf_dict = {}
	weight_list = []
	input_file = output_root_dir + project_file + '_tf.json'
	with open(input_file, 'r') as f:
		tf_list = json.load(f)
		for tf_item in tf_list:
			weight_item = {}
			term_size = sum(tf_item.values())
			for word in tf_item:
				tf = math.log(tf_item[word] / term_size + 1)
				# tf = math.log(tf_item[word]) + 1
				if word in idf_dict:
					idf = idf_dict[word]
				else:
					idf = generate_idf(word, tf_list)
					idf_dict[word] = idf
				weight_item[word] = tf * idf
			weight_list.append(weight_item)

			logger.info(project_file + "    " + str(temp) + " / " + str(total_files[project_file]))
			temp += 1

	weight_file = output_root_dir + project_file + '_w.json'
	with open(weight_file, 'w') as f:
		weight_list = json.dumps(weight_list)
		f.write(weight_list)

	logger.info(project_file + "calculate tf-idf end")


if __name__ == '__main__':
	# generate the tf-idf weight for file
	logger = get_logger('tf-idf_calculating', '../datasets/SourceFile_pre/w_log.log')

	for file in files:
		thread = TFIDFThread(file, logger)
		thread.setName(file)
		thread.start()

	# generate the tf of each term for file
	# if not os.path.exists(output_root_dir):
	# 	os.makedirs(output_root_dir)
	#
	# logger = get_logger('tf_calculating', '../datasets/SourceFile_pre/tf_log.log')
	#
	# for file in files:
	# 	thread = TFThread(file, logger)
	# 	thread.setName(file)
	# 	thread.start()




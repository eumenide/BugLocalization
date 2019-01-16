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


total_files = {'aspectj': 2394, 'eclipseUI': 17809, 'jdt': 16302, 'swt': 8560, 'tomcat': 2567}

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


class MyThread(threading.Thread):

	def __init__(self, project_file, logger):
		threading.Thread.__init__(self)
		self.project_file = project_file
		self.logger = logger

	def run(self):
		self.logger.info('Calculating tf')
		tf_thread(self.project_file, self.logger)


def get_logger():
	logger = logging.getLogger("tf_calculating")
	logger.setLevel(logging.INFO)

	fh = logging.FileHandler('tf_calculating.log')
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


if __name__ == '__main__':
	# input_file = '../datasets/SourceFile_trim/eclipseUI/eclipse.platform.ui/bundles/org.eclipse.core.commands/src/org/eclipse/core/commands/0d30649 Command.java'
	# output_file = './tmp.txt'

	if not os.path.exists(output_root_dir):
		os.makedirs(output_root_dir)

	logger = get_logger()

	for file in files:
		thread = MyThread(file, logger)
		thread.setName(file)
		thread.start()


	# for file in files:
	# 	counts = []
	# 	temp = 1
	# 	with open(input_root_dir + file + '.txt', 'r') as f:
	# 		input_files = f.readlines()
	# 		with open(output_root_dir + file + '.txt', 'a') as f2:
	# 			for input_file in input_files:
	# 				input_file = input_file.strip('\n')
	# 				output_file = re.sub(r'(.java)', '.txt', input_file)
	# 				f2.write(output_file + '\n')
	# 				input_file = input_root_dir + file + '/' + input_file
	# 				output_file = output_root_dir + file + '/' + output_file
	# 				counts.append(generate_tf(input_file, output_file))
	#
	# 				logging.info(file + "    " + str(temp) + " / " + str(total_files[file]))
	# 				temp += 1
	#
	# 	tf_file = output_root_dir + file + '_tf.json'
	# 	with open(tf_file, 'w') as f:
	# 		counts = json.dumps(counts)
	# 		f.write(counts)
	#
	# 	print('project ' + file + ' end')

	# corpus = [
	# 	['this', 'is', 'the', 'first', 'document'],
	# 	['this', 'is', 'the', 'second', 'second', 'document'],
	# 	['and', 'the', 'third', 'one'],
	# 	['is', 'this', 'the', 'first', 'document']
	# ]
	#
	# counts = Counter()
	#
	# for i in range(len(corpus)):
	# 	count = Counter(corpus[i])
	# 	print(count)
	# 	counts += count
	#
	# print(counts)



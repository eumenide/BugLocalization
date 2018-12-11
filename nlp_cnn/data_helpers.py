#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import re
import xlrd
from xml.dom import minidom


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST
	Original taken from https://github.com/yoonkim/CNN_Sentence/blob/master/process_data.py
	:param string:
	:return:
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\.", " . ", string)
	string = re.sub(r"\(", " ( ", string)
	string = re.sub(r"\)", " ) ", string)
	string = re.sub(r"\?", " ? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def load_data_from_xsl(file_name):
	"""
	Loads bug reports data from  xsl files, splits the data into words
	:param file_name:
	:return: split sentences
	"""
	# Load data from files
	data_book = xlrd.open_workbook(file_name, encoding_override='utf-8')
	data_sheet = data_book.sheet_by_index(0)

	nlp_data = []

	# Split by words
	nlp_rows = data_sheet.nrows
	for row in range(1, nlp_rows):
		nlp_data.append(clean_str(data_sheet.cell_value(row, 2) + data_sheet.cell_value(row, 3)))

	return nlp_data


def load_data_from_xml(file_name):
	"""
	Loads bug reports data from  xml files, splits the data into words
	:param file_name:
	:return: split sentences
	"""
	# Load data from files
	dom = minidom.parse(file_name)
	root = dom.documentElement
	tables = root.getElementsByTagName('database')

	nlp_data = []

	# Split by words
	for table in tables:
		columns = table.getElementsByTagName('column')
		nlp_data.append(clean_str(columns[2] + columns[3]))

	return nlp_data


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	:param data:
	:param batch_size:
	:param num_epochs:
	:param shuffle:
	:return:
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffle_data = data[shuffle_indices]
		else:
			shuffle_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffle_data[start_index:end_index]


if __name__ == '__main__':
	nlp_data = load_data_from_xsl("../datasets/AspectJ/AspectJ_1.xlsx")
	print(nlp_data[3])
	print("--------------------")
	print(nlp_data[4])
	print(nlp_data[5])
	print(nlp_data[6])
	print(nlp_data[7])
	print(nlp_data[8])
	print(nlp_data[9])


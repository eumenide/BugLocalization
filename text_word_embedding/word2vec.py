#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gensim, logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				yield line.split()


if __name__ == '__main__':
	sentences = ["Human machine interface for lab abc computer applications",
	             "A survey of user opinion of computer system response time",
	             "The EPS user interface management system",
	             "System and human system engineering testing of EPS",
	             "Relation of user perceived response time to error measurement",
	             "The generation of random binary unordered trees",
	             "The intersection graph of paths in trees",
	             "Graph minors IV Widths of trees and well quasi ordering",
	             "Graph minors A survey"]

	model = gensim.models.Word2Vec(sentences)

	word_vectors = model.wv

	print(word_vectors.vectors)

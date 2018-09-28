#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow import Variable


# 定义网络的结构
class TextCNN(object):
	"""
	A CNN for text features extraction
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""

	def __init__(self, sequence_length, vocab_size,
	             embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
		"""
		:param sequence_length: 句子固定长度（不足补全，超过截断）
		:param vocab_size: 语料库的词典大小，记为|D|
		:param embedding_size: 将词向量的维度，由原始的|D|降维到embedding_size
		:param filter_sizes: 卷积核尺寸
		:param num_filters:卷积核数量
		:param l2_reg_lambda:正则化系数
		"""
	# 变量input_x存储句子矩阵，宽为sequence_length，长度自适应（=句子数量）；
	# 变量dropout_keep_prob存储dropout参数，常量l2_loss为L2正则超参数

	# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

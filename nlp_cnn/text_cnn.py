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

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		""""
		通过一个隐藏层，将one-hot编码的词投影到一个低维空间中。
		特征提取器，在指定维度中编码语义特征，这样，语义相近的词，它们的欧氏距离或余弦举例也比较近。
		self.W可以理解为词向量词典，存储vocab_size个大小为embedding_size的词向量，随机初始化为-1~1之间的值；
		self.embedded_chars是输入input_x对应的词向量表示；
		size:[句子数量, sequence_length, embedding_size];
		self.embedded_chars_expanded是将词向量表示扩充一个维度(embedded_chars * 1)，维度变为[句子数量, sequence_length, embedding_size, 1]，方便进行卷积(tf.nn.conv2d的input参数为四维变量);
		tf.expand_dims(input, axis=None, name=None, dim=None)：在input第axis位置增加一个维度，(dim用法等同于axis，官方文档已弃用); 
		"""
		# Embedding layer 词向量层 将词组装成低维度的向量
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			self.W = tf.Variable(
				tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
				name="W")
			self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
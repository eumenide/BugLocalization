#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from nlp_cnn import data_helpers
from nlp_cnn.text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", -1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("xls_data_file", "../datasets/AspectJ/AspectJ_1.xlsx", "Data source for the xls file.")
tf.flags.DEFINE_string("xml_data_file", "../datasets/AspectJ/AspectJ.xml", "Data source for the xml file.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding(default: 128)")
tf.flags.DEFINE_string("filter_size", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probality (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess():
	# Data Preparation
	# =================================================

	# Load data
	print("Loading data...")
	x_text = data_helpers.load_data_from_xsl(FLAGS.xls_data_file)

	# Build vocabulary
	max_document_length = max([len(x.split(" ")) for x in x_text])
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x = np.array(list(vocab_processor.fit_transform(x_text)))

	return x, vocab_processor


def train(x_train, vocab_processor):
	# Training
	# =================================================

	return


def main(argv=None):
	print("jjjj")


# x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
# train(x_train, y_train, vocab_processor, x_dev, y_dev)


if __name__ == '__main__':
	# tf.app.run()
	print(preprocess())

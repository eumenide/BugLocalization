#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

root_dir = '../datasets/SourceFile/sourceFile_'
sec_dir = ['aspectj', 'eclipseUI', 'jdt', 'swt', 'tomcat']

files = []


def get_file(file_path):
	file_names = os.listdir(file_path)
	for file in file_names:
		new_dir = file_path + '/' + file
		if os.path.isfile(new_dir):
			files.append(new_dir)
		else:
			get_file(new_dir)


if __name__ == '__main__':
	for direc in sec_dir:
		get_file(root_dir + direc)
		with open(root_dir + direc + '.txt', 'w') as f:
			for file in files:
				f.write(file + '\n')
		with open(root_dir + 'count.txt', 'a') as f:
			f.write(direc + '\t' + str(len(files)) + '\n')

		print(len(files))
		files.clear()


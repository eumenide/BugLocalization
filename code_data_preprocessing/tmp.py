#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re


if __name__ == '__main__':
	string = 'JFrameWork'
	string = re.sub(r"([A-Za-z0-9][a-z])([A-Z])", lambda x: x.group(1) + " " + x.group(2), string)
	# string = re.sub(r"([A-Za-z0-9][a-z])([A-Z][a-z])", lambda x: x.group(1) + " " + x.group(2), string)
	string = re.sub(r"([A-Za-z0-9][a-z])([A-Z])", lambda x: x.group(1) + " " + x.group(2), string)
	print(string.strip())
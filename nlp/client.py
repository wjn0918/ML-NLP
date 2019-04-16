#!/usr/bin/env python
# encoding: utf-8

import hprose

def main():
	client = hprose.HttpClient('http://127.0.0.1:8181/')
	with open('lib/cs.txt', encoding='utf8') as rd:
		doc = rd.read()
	# doc = input("请输入：")
	print(doc)
	import time
	time.sleep(3)
	print(client.pre_txt(doc))

if __name__ == '__main__':
	main()
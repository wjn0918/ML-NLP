import hprose

from nlp.demo import action
from nlp.tools.db import conn_mongodb


def pre_txt(doc):
	"""
	预测文本类别
	:param doc: 需要分类的文本
	:return:
	"""
	r = action(doc)

	return r

def main():
	server = hprose.HttpServer(port = 8181)
	server.addFunction(pre_txt)
	server.start()

if __name__ == '__main__':
	main()

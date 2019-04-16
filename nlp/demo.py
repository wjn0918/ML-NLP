"""训练模型"""
import logging
import os

from numpy.ma import zeros

from jieba import analyse
from jieba import posseg as pseg
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from gensim import corpora, models, similarities

from nlp.tools.db import executeSql, conn_mongodb
from nlp.tools.preprocess import tokenization

from configparser import ConfigParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def setWord2Vec(vocabList, inputSet):
    """
    词语映射成矩阵
    类似[(index_1,number_1),(index_2,number_2)]
    index-----词语在整个词语去重列表（vocabList）的角标
    number----词语出现的次数
    """
    rNum = len(inputSet)
    cNum = len(vocabList)
    returnVec = zeros((rNum, cNum))
    for i in range(rNum):
        # print(inputSet[i])
        for j in range(len(inputSet[i])):
            # print(inputSet[i][j], end = "   ")
            try:
                r = vocabList.index(inputSet[i][j])
                returnVec[i][r] = 1
            except ValueError:
                continue
    # print(returnVec)
    return returnVec


def createDataList(dataArr):
    """
    去重词语列表，创建词袋
    :param dataArr:
    :return:
    """
    returnDataList = set()
    for data in dataArr:
        returnDataList = returnDataList | set(data)
    # print(returnDataList)
    return list(returnDataList)


def setWord2Vec(vocabList, inputSet):
    """
    构建矩阵模型
    :param vocabList: 全量词语
    :param inputSet: 需要构建的文章
    :return:
    """

    rNum = len(inputSet)
    cNum = len(vocabList)
    returnVec = zeros((rNum, cNum))
    for i in range(rNum):
        # print(inputSet[i])
        for j in range(len(inputSet[i])):
            # print(inputSet[i][j], end = "   ")
            try:
                r = vocabList.index(inputSet[i][j])
                returnVec[i][r] = 1
            except ValueError:
                continue
    # print(returnVec)
    return returnVec


def get_data():
    """
    从数据库中加载数据
    """
    dataSet = {"data": [], "label": []}
    cf = ConfigParser()
    cf.read('conf/sql.ini')
    sql = cf.get('load_data', 'sql')
    r = executeSql(sql, returnDict=True)
    for data in r:
        dataSet['data'].append(data['doc'])
        dataSet['label'].append(data['label'])
    # print(dataSet)
    return dataSet
    pass


def compute_similarity(category, article):
    """
    计算文本之间的相似性
    @:param category : 需要比对的文本类别
    @:param article : 需要比对的文本
    @:return 与文本库中文本之间的相似度
    """
    results = []
    all_articles = get_similarity_article(category)
    reduced_contents = []
    for i in all_articles:
        reduced_contents.append(tokenization(i['doc']))
    print(reduced_contents)
    dictionary = corpora.Dictionary(reduced_contents)
    #生成词袋模型
    corpus = [dictionary.doc2bow(text) for text in reduced_contents]
    tfidf = models.TfidfModel(corpus)
    vec = dictionary.doc2bow(tokenization(article))
    index = similarities.MatrixSimilarity(tfidf[corpus])  # 对整个语料库进行转换并编入索引，准备相似性查询
    sorted_r = sorted(list(enumerate(index[vec])),key=lambda x:x[1],reverse=True)
    for r in sorted_r:
        results.append(all_articles[r[0]]['doc']+"距离为："+str(r[1]))
        print(all_articles[r[0]]['doc']+"距离为："+str(r[1]))
    return results


def get_similarity_article(category):
    """
    获取相似类别的文章
    :param category: 文章类别
    :return: 相似类别的文章数组
    """
    sql = 'select doc from t_news where label = %s'
    args = (category)
    return executeSql(sql, args=args, returnDict=True)


def constructWordVec(txts):
    """
    将文档数组构建成词向量 例：[[(1,3),(2,4)],
                              [(1,2),(2,2)]]
    :param txts: 多个文档构成的数组  例：[['我','爱','你'],
                                          ['我','喜欢','你']]
    :return:词向量构成的数组
    """
    print(txts)
    dictionary = corpora.Dictionary(txts)
    return [dictionary.doc2bow(text) for text in txts]


def writeLmk(fileName, landmarks):
    """将list数据保存"""
    fp = open(fileName, 'w+')
    fp.write(
        "version: 1" + '\n'
        "n_points: 68" + '\n'
        "{" + '\n'
    )
    for i in range(len(landmarks)):
        fp.write(str(landmarks[i][1]))
        fp.write(" ")
        fp.write(str(landmarks[i][0]) + '\n')

    fp.write("}")
    fp.close()
    return True


def preCategory(txt):
    """
    预测文本的类别
    :param txt: 需要分类的文本，String
    :return: 文本类别 String
    """
    datas = get_data()
    contents = datas['data']
    labels = datas['label']
    reduced_contents = []
    for content in contents:
        reduced_contents.append(tokenization(content))
    all_word = createDataList(reduced_contents)
    # print(all_word)
    # print(type(all_word))
    mnb = MultinomialNB()
    dataVec = setWord2Vec(all_word, reduced_contents)
    mnb.fit(dataVec, labels)
    r = [tokenization(txt)]
    csDataVec = setWord2Vec(all_word, r)
    category = mnb.predict(csDataVec)[0]
    return category

def action(doc):
    dataSets = {"label":"","simi_doc":[]}
    import datetime
    s = datetime.datetime.now()
    # with open('lib/cs.txt', encoding='utf8') as rd:
    #     doc = rd.read()
    category = preCategory(doc)
    dataSets['label'] = category
    print(category)
    doc_score = compute_similarity(category, doc)
    dataSets["simi_doc"] = doc_score
    print((datetime.datetime.now() - s).seconds)

    return dataSets

def cs():
    import datetime
    s = datetime.datetime.now()
    with open('lib/cs.txt', encoding='utf8') as rd:
        doc = rd.read()
    category = preCategory(doc)
    print(category)
    doc_score = compute_similarity(category, doc)
    print((datetime.datetime.now() - s).seconds)

if __name__ == '__main__':
    cs()

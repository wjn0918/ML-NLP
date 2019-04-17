"""主程序"""
import datetime
import logging
import os
import pickle

import pypinyin
from numpy.ma import zeros

from sklearn.externals import joblib
from gensim import corpora, models, similarities

from nlp.tools.ModelSelection import Models
from nlp.tools.db import executeSql
from nlp.tools.preprocess import tokenization

from configparser import ConfigParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def hp(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


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
    fr1 = open('lib/dump_cateVec_%s.txt' %(hp(category)), 'rb', buffering=0)
    index = pickle.load(fr1)

    fr2 = open('lib/dump_fenci_%s.txt' % (hp(category)), 'rb', buffering=0)
    dictionary = pickle.load(fr2)

    results = []
    all_articles = get_similarity_article(category)
    # reduced_contents = []
    # for i in all_articles:
    #     reduced_contents.append(tokenization(i['doc']))
    # print(reduced_contents)
    # dictionary = corpora.Dictionary(reduced_contents)
    #生成词袋模型
    # corpus = [dictionary.doc2bow(text) for text in reduced_contents]
    # tfidf = models.TfidfModel(corpus)
    vec = dictionary.doc2bow(tokenization(article))
    # index = similarities.MatrixSimilarity(tfidf[corpus])  # 对整个语料库进行转换并编入索引，准备相似性查询
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




def init():
    """
    初始化模型，数据集（每天12点进行）
    :return:
    """
    print('Init! The time is: %s' % datetime.datetime.now())
    datas = get_data()
    contents = datas['data']
    labels = datas['label']


    # 将分词后的文本序列化到本地
    reduced_contents = []
    for content in contents:
        reduced_contents.append(tokenization(content))
    f1 = open('lib/dump_fenci.txt', 'wb', buffering=0)
    pickle.dump(reduced_contents, f1)
    f1.close()


    all_word = createDataList(reduced_contents)


    # 将词袋序列化到本地
    f2 = open('lib/dump.txt', 'wb', buffering=0)
    pickle.dump(all_word, f2)
    f2.close()


    #将词向量序列化到本地
    f3 = open('lib/dump_wordVec.txt', 'wb', buffering=0)
    dataVec = setWord2Vec(all_word, reduced_contents)
    pickle.dump(dataVec, f3)
    f3.close()


    # 将训练好的模型序列化到本地
    model = Models.mnb.value
    model.fit(dataVec, labels)
    joblib.dump(model, 'lib/mnb.m')


    # 将各类别文章词袋模型存入本地
    sql = 'select label from t_news group by label'
    labels = executeSql(sql, returnDict=True)
    for label in labels:
        cate = label['label']
        all_articles = get_similarity_article(cate)
        reduced_contents = []
        for i in all_articles:
            reduced_contents.append(tokenization(i['doc']))
        dictionary = corpora.Dictionary(reduced_contents)
        f4 = open('lib/dump_fenci_%s.txt' % (hp(cate)), 'wb', buffering=0)
        pickle.dump(dictionary, f4)
        f4.close()
        # 生成词袋模型
        corpus = [dictionary.doc2bow(text) for text in reduced_contents]
        tfidf = models.TfidfModel(corpus)
        index = similarities.MatrixSimilarity(tfidf[corpus])  # 对整个语料库进行转换并编入索引，准备相似性查询
        f5 = open('lib/dump_cateVec_%s.txt' %(hp(cate)), 'wb', buffering=0)
        pickle.dump(index, f5)
        f5.close()


    print('Init sucessfull! The time is: %s' % datetime.datetime.now())



def preCategory(txt):
    """
    预测文本的类别
    :param txt: 需要分类的文本，String
    :return: 文本类别 String
    """
    fr = open('lib/dump.txt', 'rb')
    all_word = pickle.load(fr)
    mnb = joblib.load('lib/mnb.m')
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
    """
    仅用于测试
    """
    with open('lib/cs.txt', encoding='utf8') as rd:
        doc = rd.read()
    category = preCategory(doc)
    print(category)


if __name__ == '__main__':
    cs()

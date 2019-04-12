"""训练模型"""
import jieba
import pymysql
from numpy.ma import zeros

import re
import functools
from jieba import analyse
from jieba import posseg as pseg
from pymysql.cursors import DictCursor
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from gensim import corpora, models, similarities


def createDataList(dataArr):
    """创建词袋"""
    returnDataList = set()
    for data in dataArr:
        returnDataList = returnDataList | set(data)
    # print(returnDataList)
    return list(returnDataList)

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


def tokenization(content):
    '''
    {标点符号、连词、助词、副词、介词、时语素、‘的’、数词、方位词、代词}
    {'x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r'}
    去除文章中特定词性的词
    :content str
    :return list[str]
    '''
    # stop_flags = {'x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r'}
    stop_flags = {'x'}
    stop_words = {'了'}
    words = pseg.cut(content)
    words = ''.join([word for word, flag in words if flag not in stop_flags and word not in stop_words])
    return analyse.extract_tags(words,10)

def createDataList(dataArr):
    """
    去重词语列表
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


def get_db():
    host = 'localhost'
    port = 3306
    user_name = 'root'
    password = '123456'
    db_name = 'cs'
    conn = pymysql.connect(
        host,
        user_name,
        password,
        db_name)
    return conn

def get_data():
    """
    从数据库中加载数据
    """
    dataSet = {"data":[],"label":[]}
    cursor = get_db().cursor(DictCursor)
    sql = "select content,label from t_txt"
    cursor.execute(sql)
    for data in cursor.fetchall():
        dataSet['data'].append(data['content'])
        dataSet['label'].append(data['label'])
    # print(dataSet)
    return dataSet
    pass

def compute_similarity(all_articles,article):
    """
    计算文本之间的相似性
    """
    reduced_contents = []
    for i in all_articles:
        reduced_contents.append(tokenization(i))
    dictionary = corpora.Dictionary(reduced_contents)
    #生成词袋模型
    corpus = [dictionary.doc2bow(text) for text in reduced_contents]
    # print(corpus)
    tfidf = models.TfidfModel(corpus)
    vec = dictionary.doc2bow(tokenization(article))
    index = similarities.MatrixSimilarity(tfidf[corpus])  # 对整个语料库进行转换并编入索引，准备相似性查询
    sorted_r = sorted(list(enumerate(index[vec])),key=lambda x:x[1],reverse=True)
    # print(all_articles)
    for r in sorted_r:
        print(all_articles[r[0]]+"\tscore: " + str(r[1]))


def get_similarity_article(category,article):
    sql = 'select content from t_txt where label = "%s"' %(category)
    cursor = get_db().cursor(DictCursor)
    cursor.execute(sql)
    all_articles = []
    for r in cursor.fetchall():
        all_articles.append(r['content'])
    cursor.close()
    compute_similarity(all_articles,article)
    pass


if __name__ == '__main__':

    datas = get_data()
    contents = datas['data']
    labels = datas['label']
    reduced_contents = []
    for content in contents:
        reduced_contents.append(tokenization(content))
    global all_word
    all_word = createDataList(reduced_contents)
    # print(reduced_contents)
    mnb = MultinomialNB()
    dataVec = setWord2Vec(all_word, reduced_contents)
    mnb.fit(dataVec, labels)
    csData = '他身价高达千亿为博红颜一笑狂撒10亿买房现68岁竟悔不当初'
    while True:
        csData = input("请输入需要检索的文章：")
        r = [tokenization(csData)]
        csDataVec = setWord2Vec(all_word, r)
        s = mnb.predict(csDataVec)[0]
        print("该文章属于:%s" %(s))
        import time
        time.sleep(3)
        print("类似文章有：")
        get_similarity_article(s,csData)
    
    


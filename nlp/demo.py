"""训练模型"""
import logging

from numpy.ma import zeros

from jieba import analyse
from jieba import posseg as pseg
from sklearn.naive_bayes import MultinomialNB
from gensim import corpora, models, similarities

from nlp.tools.db import executeSql
from nlp.tools.preprocess import tokenization


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
    dataSet = {"data":[],"label":[]}
    sql = "select content,label from t_nlp"
    r = executeSql(sql,returnDict=True)
    for data in r:
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
    tfidf = models.TfidfModel(corpus)
    vec = dictionary.doc2bow(tokenization(article))
    index = similarities.MatrixSimilarity(tfidf[corpus],num_features=12)  # 对整个语料库进行转换并编入索引，准备相似性查询
    sorted_r = sorted(list(enumerate(index[vec])),key=lambda x:x[1],reverse=True)
    # print(all_articles)
    for r in sorted_r:
        print(all_articles[r[0]]+"\tscore: " + str(r[1]))



def get_similarity_article(category,article):
    """
    获取相似类型文章
    :param category: 文章分类
    :param article: 需要查找类型的文章
    :return: 相似类型文章
    """
    sql = 'select content from t_nlp where label = %s'
    args = (category)
    results = executeSql(sql,args=args,returnDict=True)
    if len(results) != 0:
        all_articles = []
        for r in results:
            all_articles.append(r['content'])
        compute_similarity(all_articles,article)

    pass

def constructWordVec(txts):
    print(txts)
    dictionary = corpora.Dictionary(txts)
    return [dictionary.doc2bow(text) for text in txts]


def preCategory(txt):
    """
    预测文本的类别
    :param txt: 需要分类的文本
    :return: 文本类别
    """
    datas = get_data()
    contents = datas['data']
    labels = datas['label']
    reduced_contents = []
    # 将每篇文章的所有权重前n词放入数组中
    for content in contents:
        reduced_contents.append(tokenization(content))
    all_word = createDataList(reduced_contents)
    mnb = MultinomialNB()
    dataVec = setWord2Vec(all_word, reduced_contents)
    mnb.fit(dataVec, labels)
    r = [tokenization(txt)]
    csDataVec = setWord2Vec(all_word, r)
    s = mnb.predict(csDataVec)[0]
    print("该文章属于:%s" % (s))
    return s

def main():

    csData = '他身价高达千亿为博红颜一笑狂撒10亿买房现68岁竟悔不当初'
    category = preCategory(csData)
    get_similarity_article(category,csData)



if __name__ == '__main__':
    main()

    


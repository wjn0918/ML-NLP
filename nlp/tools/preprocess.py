"""预处理"""
from jieba import posseg as pseg
from jieba import analyse
from gensim import corpora
import os

# 数据标准化
def tokenization(content):
    '''
    {形容词（a）、连词(c)、副词(d)、叹词（e）、方位词(f)、数词(m)、量词（q）、标点符号(w)、时间词（t）}
    去除文章中特定词性的词
    :content 需要标准化的数据
    :return list[str]
    '''
    # stop_flags = {'x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r'}
    stop_flags = {'x','q','w','t','m'}              #除去对应词性的词
    words = pseg.cut(content)
    words = ''.join([word for word, flag in words if flag not in stop_flags])
    # stop_words_path = os.path.split(os.getcwd())[0] + '/lib/stop_words.txt'    #加载停用词
    stop_words_path = 'lib/stop_words.txt'
    analyse.set_stop_words(stop_words_path)         #除去停用词
    return analyse.extract_tags(words)


def consturctWordVec(texts):
    """
    构建词向量
    :param text: 所有文章构成的数组
    :return:
    """

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(corpus)
if __name__ == '__main__':
    print(tokenization("在得到每一篇文档对应的主题向量后，我们就可以计算文档之间的相似度，进而完成如文本聚类、信息检索之类的任务。在Gensim中，也提供了这一类任务的API接口。"))
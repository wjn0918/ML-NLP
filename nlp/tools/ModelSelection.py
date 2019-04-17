"""模型选择"""

from enum import Enum
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from configparser import ConfigParser


class Models(Enum):
    cf = ConfigParser()
    cf.read('conf/model.ini')
    items = dict(cf.items("alpha"))
    __alpha_mnb = float(items['mnb'])
    __alpha_bnb = float(items['bnb'])
    # __alpha_gnb = float(items['gnb'])
    __alpha_cnb = float(items['cnb'])

    mnb = MultinomialNB(__alpha_mnb)#多项式朴素贝叶斯模型
    bnb = BernoulliNB(__alpha_bnb)  # 伯努利朴素贝叶斯模型
    gnb = GaussianNB()  # 高斯朴素贝叶斯模型
    cnb = ComplementNB(__alpha_cnb)  # 补充朴素贝叶斯模型
    pass

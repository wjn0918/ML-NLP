"""
使用连接池连接数据库，执行sql

"""
import pymysql
from DBUtils.PooledDB import PooledDB

from configparser import ConfigParser
#读取配置
from pymysql.cursors import DictCursor

cf = ConfigParser()
cf.read('conf/db.conf')


def conn_mysql():
    host = cf.get('mysql','host')
    user_name = cf.get('mysql','user_name')
    password = cf.get('mysql', 'password')
    db = cf.get('mysql', 'db')
    pool = PooledDB(pymysql, 5, host=host, user=user_name, passwd=password, db=db, port=3306, setsession=['SET AUTOCOMMIT = 1'])
    # 5为连接池里的最少连接数，setsession=['SET AUTOCOMMIT = 1']是用来设置线程池是否打开自动更新的配置，0为False，1为True
    conn = pool.connection()
    return conn


def executeSql(sql,args=None,returnDict=False):
    """
    执行sql
    :param sql:sql 语句
    :param args: sql 中的参数
    :param returnDict: 是否创建返回字典类型游标
    :return: 查询的所有结果
    """
    conn = conn_mysql()
    if returnDict:
        cursor = conn.cursor(DictCursor)
    else:
        cursor = conn.cursor()
    cursor.execute(sql, args)
    r = cursor.fetchall()
    conn.commit()
    cursor.close()
    conn.close()
    return r
    pass

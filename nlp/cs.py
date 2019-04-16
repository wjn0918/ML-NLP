try:
    import cPickle as pickle
except ImportError:
    import pickle

obj = ['张三','李四','王五']
f = open('dump.txt','wb')
pickle.dump(obj,f)
f.close()

f = open('dump.txt','rb')
r = pickle.load(f)
print(r)
print(type(r))
f.close()
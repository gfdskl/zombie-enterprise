#-*- coding : utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import seaborn as sns
from sklearn.cluster import KMeans

from settings import Settings
from load_data import loadData
from preprocess import preProcess
from pulearning import PULlearning
import OTO
import model_eval as me

# 导入训练集和验证集数据
dir = 'process_data'
TRAIN_DIR = os.path.join(dir,'train_process.csv')
VERIFY_DIR = os.path.join(dir,'verify_process.csv')

if not os.path.exists(TRAIN_DIR):
    train,verify = OTO.preProcess_train_verify()
else:
    train,verify = pd.read_csv(TRAIN_DIR),pd.read_csv(VERIFY_DIR)


# 训练并保存模型
pu = PULlearning(train)
classifier = pu.Spy()


# 使用验证集评估
verify_x = verify.iloc[:,:-1].values
verify_y = verify.iloc[:,-1].values
pred_verify_y = classifier.predict(verify_x)
pred_verify_y[np.argwhere(pred_verify_y==-1)] = 0

verify_y[np.argwhere(verify_y==-1)] = 0
me.evalModel(verify_y,pred_verify_y)






















# train,verify = train.values,verify.values


# kmeans = KMeans(n_clusters=1)

# x = union_table.iloc[:,:-1].values
# y = union_table.iloc[:,-1].values

# print (y)
# x = x[np.argwhere(y==1).reshape(-1,)]
# print (x.shape)

# kmeans.fit(x)
# pred_y = kmeans.predict(x)
# print (pred_y)


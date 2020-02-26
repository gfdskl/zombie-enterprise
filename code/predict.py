import os
import glob
import pickle
import numpy as np
import pandas as pd

from preprocess import preProcess


dtypes = [
    {'ID':"Int64",'注册时间':"Int64",'注册资本':np.float64,'行业':object,'区域':object,'企业类型':object,'控制人类型':object,'控制人持股比例':np.float64,'flag':"Int64"},
    "Int64",
    {'ID':"Int64",'year':"Int64",'债权融资额度':np.float64,'债权融资成本':np.float64,'股权融资额度':np.float64,'股权融资成本':np.float64,'内部融资和贸易融资额度':np.float64,'内部融资和贸易融资成本':np.float64,'项目融资和政策融资额度':np.float64,'项目融资和政策融资成本':np.float64},
    {'ID':"Int64",'year':np.float64,'从业人数':np.float64,'资产总额':np.float64,'负债总额':np.float64,'营业总收入':np.float64,'主营业收入':np.float64,'利润总额':np.float64,'净利润':np.float64,'纳税总额':np.float64,'所有者权益合计':np.float64}]

model_path = "/Users/sameal/Documents/PROJECT/zombie_enterprise/code/model/XGBClassifier_pd.pickle"

def get_file_path(dir_path):
    base_path = glob.glob(os.path.join(dir_path, "*base*.csv"))[0]
    knowledge_path = glob.glob(os.path.join(dir_path, "*knowledge*.csv"))[0]
    money_path = glob.glob(os.path.join(dir_path, "*money*.csv"))[0]
    year_path = glob.glob(os.path.join(dir_path, "*year*.csv"))[0]
    return (base_path, knowledge_path, money_path, year_path)


def load_data(dir_path):
    file_path = get_file_path(dir_path)
    data = []
    for i in range(4):
        table = pd.read_csv(file_path[i], encoding='gbk', dtype=dtypes[i])
        data.append(table)
    return data


def preprocess_data(data: [pd.DataFrame]):
    pp = preProcess(data, saveName=None)
    union_table = pp.precoss()
    return union_table


def predict(dir_path):
    data = load_data(dir_path)
    union_table = preprocess_data(data)
    union_table = union_table.drop(columns="flag")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    pred_flag = model.predict(union_table)
    full_table = pd.concat([union_table, pd.Series(pred_flag, index=union_table.index, name="pred_flag")], axis=1)
    return full_table


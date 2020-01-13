#-*- coding : utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.preprocessing import MinMaxScaler

class preProcess():
    def __init__(self,data,saveName='train'):
        super().__init__()
        self.t1 = data[0]
        self.t2 = data[1]
        self.t3 = data[2]
        self.t4 = data[3]
        self.saveName = saveName


    def precoss(self):
        # 数据预处理，包括缺失值填补、归一化和独热编码
        union_table = self.unionAllTable()
        # print (union_table.shape)
        # print (union_table.columns)
        # print (union_table.isna().sum())
        union_table = self.fillNa(union_table)
        # print (union_table.head())
        onehot_feature = ['行业','区域','企业类型','控制人类型','专利','商标','著作权']
        union_table = self.oneHot(union_table,onehot_feature)
        scale_features = list(set(union_table.columns)-set(['ID','flag']))
        union_table = self.minMaxScaler(union_table,scale_features)

        columns_num = union_table.shape[1]
        union_table.insert(columns_num-1,'flag',union_table.pop('flag'))

        union_table_id = union_table['ID']
        del union_table['ID']
        self.saveData(union_table,union_table_id)
        return union_table

    def unionAllTable(self):
        # 四表融合
        process_t3 = self.getUnionAll(self.t3)
        process_t4 = self.getUnionAll(self.t4)

        tmp1 = pd.merge(self.t1,self.t2,on='ID')
        tmp2 = pd.merge(process_t3,process_t4,on='ID')
        union_table = pd.merge(tmp1,tmp2,on='ID',how='left')
        return union_table


    def getData_15_16_17(self,data):
        # 得到15，16，17年数据
        data_2015 = data[data.year == 2015]
        data_2016 = data[data.year == 2016]    
        data_2017 = data[data.year == 2017] 
        
        data_2015 = data_2015.set_index('ID',drop=False)
        data_2016 = data_2016.set_index('ID',drop=False)    
        data_2017 = data_2017.set_index('ID',drop=False)  
        return data_2015,data_2016,data_2017
    
    def getDiffValue(self,past,present,prefix):
        # 得到前后两年差值
        diff_value = present-past
        diff_value.drop(columns=['ID','year'],inplace=True)
        diff_value.rename(columns=lambda x:prefix+x+'差额',inplace=True)    # 给列名添加前缀prefix和后缀'差额'
        diff_value['ID'] = diff_value.index
        diff_value.index.name = 'id'
        return diff_value

    def getAverage(self,data):
        # 得到三年平均值
        ans = data.groupby('ID').sum()/3
        ans.drop(columns=['year'],inplace=True)
        ans.rename(columns=lambda x:x+'平均值',inplace=True)
        ans['ID'] = ans.index
        ans.index.name = 'id'
        return ans
        

    def getUnionAll(self,data):
        # 将两张差额表和平均值表连在一起得到新表
        data_2015,data_2016,data_2017 = self.getData_15_16_17(data)
        diff_value1 = self.getDiffValue(data_2015,data_2016,'2015-2016')
        diff_value2 = self.getDiffValue(data_2016,data_2017,'2016-2017')
        average = self.getAverage(data)
        
        diff_value = pd.merge(diff_value1,diff_value2,on="ID")
        data_union_all = pd.merge(diff_value,average,on="ID",how="right")
        data_union_all['ID'] = data_union_all['ID'].astype(int)
        data_union_all.sort_values("ID",inplace=True)
        return data_union_all

    def minMaxScaler(self,data,scale_features):
        # 归一化
        ss = MinMaxScaler()
        data[scale_features] = ss.fit_transform(data[scale_features])
        return data

    def oneHot(self,data,onehot_feature):
        # 独热编码
        data = pd.get_dummies(data,columns=onehot_feature)
        return data

    def feature_map(self,data,feature):
        # 将离散类数据映射为int类型数据
        feature_unique = data[feature].unique()
        feature_dict = dict(zip(feature_unique,[i for i in range(len(feature_unique))]))
        data[feature] = data[feature].map(feature_dict).fillna(data[feature].mode()).astype(int)
        return data

    def fillNa(self,data):
        # 填补缺失值
        data['注册时间'] = data['注册时间'].fillna(data['注册时间'].mode()[0])
        data['flag'] = data['flag'].fillna(-1)
        data = self.feature_map(data,'行业')
        data = self.feature_map(data,'区域')
        data = self.feature_map(data,'企业类型')
        data = self.feature_map(data,'控制人类型')
        data = self.feature_map(data,'商标')    
        data = self.feature_map(data,'专利')    
        data = self.feature_map(data,'著作权')
        
        # print ('*'*100)
        # print (data.isna().sum())
        # print (data.dtypes)
        data.fillna(data.mean(),inplace=True)
        # print ('*'*100)
        # print (data)

        return data


    def saveData(self,union_table,union_table_id):
        # 保存预处理结果
        dir = 'process_data'
        if not os.path.exists(dir):
            os.mkdir(dir)
        saveDir = self.saveName+'_process.csv'
        saveDir_id = self.saveName+'_process_id.csv'
        union_table.to_csv(os.path.join(dir,saveDir),index=0,encoding="utf_8_sig")
        union_table_id.to_csv(os.path.join(dir,saveDir_id),index=0,header=True)


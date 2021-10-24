#-*- coding : utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.base import ClassifierMixin
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost

import pickle


class PULlearning():
    # Classifier = lambda classifier:classifier()

    # def Classifier(self):
    #     return self.classifier()

    def __init__(self,data,classifier):
        super().__init__()
        self.data = data
        self.classifier = classifier
        

    def Spy(self):
        # 用Spy算法找到可靠的负样本
        # if (self.classifier == 'XGBClassifier'):
        #     model_name = self.classifier+'.xgb'
        # else:
        #     model_name = self.classifier+'.pickle'
        model_name = self.classifier+'.pickle'
        model_dir = 'model'
        full_name = os.path.join(model_dir,model_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if os.path.exists(full_name):
            with open(full_name, 'rb') as fr:
                classifier = pickle.load(fr)
                # if (self.classifier == 'XGBClassifier'):
                #     print ("full_name:"+full_name)
                #     classifier = xgboost.Booster()
                #     classifier.load_model(model_name)
                #     # classifier = xgboost.load_model(fr)
                # else:
                #     classifier = pickle.load(fr)
                return classifier

        x = self.data.iloc[:,:-1].values
        y = self.data.iloc[:,-1].values

        p = np.argwhere(y==1).reshape(-1,)
        u = np.argwhere(y!=1).reshape(-1,)

        row_rand_array = np.arange(p.shape[0])
        np.random.shuffle(row_rand_array)
        num = int(p.shape[0]*0.15)
        s = row_rand_array[0:num]

        ps = list(set(p)-set(s))
        us = list(set(u)|set(s))

        PS = x[ps]
        US = x[us]
        PS_label = np.ones(PS.shape[0]).reshape(-1,1)
        US_label = (-1*np.ones(US.shape[0])).reshape(-1,1)

        new_x = np.vstack((PS,US))
        new_y = np.vstack((PS_label,US_label)).reshape(-1,)

        # classifier = XGBClassifier()
        classifier = eval(self.classifier)()
        classifier.fit(new_x,new_y)

        S = x[s]
        U = x[u]     

        pred_y_U = classifier.predict_proba(U)
        tr = self.getTr(classifier,S)
        
        rn = np.argwhere(pred_y_U[:,-1]<tr).reshape(-1,)

        RN = U[rn]
        P = x[p]
        RN_label = (-1*np.ones(RN.shape[0])).reshape(-1,1)
        P_label = np.ones(P.shape[0]).reshape(-1,1)

        new_x = np.vstack((P,RN))
        new_y = np.vstack((P_label,RN_label)).reshape(-1,)
        
        classifier.fit(new_x,new_y)
        with open(full_name, 'wb') as fw:
                # if (self.classifier == 'XGBClassifier'):
                #     # classifier.save_model(fw)
                #     pass
                # else:
                #     pickle.dump(classifier, fw)
            pickle.dump(classifier, fw)
        return classifier




    def getTr(self,classifier,S):
        pre_y_S = classifier.predict_proba(S)
        best_tr = 0

        for i in range(0,100):
            tr = i/100
            a = len(np.argwhere(pre_y_S[:,-1]>tr))/pre_y_S.shape[0]
            if a < 0.975:
                break
            best_tr = tr

        return best_tr


                




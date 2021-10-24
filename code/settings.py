import numpy as np

class Settings():
    def __init__(self):
        # 初始化参数
        super().__init__()
        self.DIR = r'E:\大学生活\比赛\服务外包\A09-科创-企业画像数据接口'
        # 使用XGBoost、随机森林、GBDT、线性回归模型
        self.MODEL = ['XGBClassifier','RandomForestClassifier','GradientBoostingClassifier','LogisticRegression']
        # self.MODEL = ['LogisticRegression']
        # self.MODEL = ['RandomForestClassifier','GradientBoostingClassifier','LogisticRegression']
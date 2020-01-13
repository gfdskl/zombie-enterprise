#-*- coding : utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn

class loadData():
    # 导入数据
    def __init__(self,dir,key):
        # 通过路径和key找到需要的文件
        super().__init__()
        self.dir = dir
        self.key = key

    def load(self):
        # 加载所有文件
        readCsv = lambda file_name,dtype:pd.read_csv(os.path.join(self.dir,file_name),encoding='gbk',dtype=dtype)
        files = os.listdir(self.dir)
        use_files = [i for i in files if self.key in i]
        if self.key == 'verify':
            use_files[1],use_files[2] = use_files[2],use_files[1]

        dtypes = list()
        base_dtype = {'ID':"Int64",'注册时间':"Int64",'注册资本':np.float64,'行业':object,'区域':object,
                            '企业类型':object,'控制人类型':object,'控制人持股比例':np.float64,'flag':"Int64"}
        knowledge_dtype = "Int64"
        money_report_dtype = {'ID':"Int64",'year':"Int64",'债权融资额度':np.float64,'债权融资成本':np.float64,                             '股权融资额度':np.float64,'股权融资成本':np.float64,                                                        '内部融资和贸易融资额度':np.float64,'内部融资和贸易融资成本':np.float64,
                                '项目融资和政策融资额度':np.float64,'项目融资和政策融资成本':np.float64}
        year_report_dtype = {'ID':"Int64",'year':np.float64,'从业人数':np.float64,'资产总额':np.float64,
                                '负债总额':np.float64,'营业总收入':np.float64,'主营业收入':np.float64,
                                '利润总额':np.float64,'净利润':np.float64,'纳税总额':np.float64,
                                '所有者权益合计':np.float64}
        dtypes.append(base_dtype)
        dtypes.append(knowledge_dtype)
        dtypes.append(money_report_dtype)
        dtypes.append(year_report_dtype)

        
        data = list()
        for i in range(4):
            table = readCsv(use_files[i],dtypes[i])
            if '控制人ID' in table:
                del table['控制人ID']
            data.append(table)

        return data
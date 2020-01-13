from load_data import loadData
from settings import Settings
from preprocess import preProcess



def load_train_verity():
    # 导入训练集和验证集
    ss = Settings()
    key = ['train','verify']
    ld1 = loadData(ss.DIR,key[0])
    ld2 = loadData(ss.DIR,key[1])
    train = ld1.load()
    verify = ld2.load()
    return train,verify

def preProcess_train_verify():
    # 预处理训练集和验证集
    saveName = ['train','verify']
    train,verify = load_train_verity()
    pp1 = preProcess(train,saveName[0])
    pp2 = preProcess(verify,saveName[1])
    pre_train,pre_verify = pp1.precoss(),pp2.precoss()
    return pre_train,pre_verify
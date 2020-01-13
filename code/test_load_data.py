from load_data import loadData
import pandas as pd
from settings import Settings


ss = Settings()
key = 'verify'
ld = loadData(ss.DIR,key)
data = ld.load()


print (type(data))
for i in range(4):
    print (data[i].head())
    print (data[i].shape)

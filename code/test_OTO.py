import OTO
import pandas as pd


PRINTX = lambda x:print ('*'*100)
train,verify = OTO.preProcess_train_verify()

print (train.head())
print (train.shape)
print (type(train))

PRINTX(100)

print (verify.head())
print (verify.shape)
print (type(verify))
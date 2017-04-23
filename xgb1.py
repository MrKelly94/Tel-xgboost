# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import xgboost as xgb
import pandas as pd
import numpy as nm

train1=pd.read_csv("data/train.csv")
label1=pd.read_csv("data/label.csv")
#train2=pd.read_csv("data/train1.csv")
label2=pd.read_csv("data/testLabel.csv")
test=pd.read_csv("data/test.csv")
#label0=pd.read_csv("label0.csv")
ids=test['id'].values

#bst<-xgboost(data=teltrain_data, label=teltrain_label, max.depth=2, eta=1, nround=2, objective="binary:logistic")
param={'objective':'multi:softprob', 'num_class':3, 'eval_metric':'mlogloss', 'booster': 'gbtree', 'colsample_bytree':0.75,
'gamma':1.5,'max_depth':25, 'eta':0.1, 'min_child_weight':0.5, 'subsample':0.8, 'nthread':4}
plst = list(param.items())
#训练集共30017，划分20000用作训练，10017用作验证
#offset=20000 
#迭代次数
num_rounds=100

telest=xgb.DMatrix(test)
#划分训练集与验证集
teltrain=xgb.DMatrix(train1, label=label1)
telval=xgb.DMatrix(test, label=label2)



watchlist=[(teltrain, 'train'), (telval, 'val')];
# Run Cross Valication
cv = xgb.cv(param, teltrain, metrics='mlogloss', num_boost_round=num_rounds, nfold=5,  seed=1, early_stopping_rounds=25,  show_stdv=False)
best_nrounds = cv.shape[0] - 1
#训练模型
model = xgb.train(plst, teltrain, best_nrounds, watchlist, )
pred = model.predict(telest)
preds=nm.matrix(pred)
print (preds)
nm.savetxt('submission9.csv', nm.c_[ids,preds], delimiter=',', header='id,predict_0,predict_1,predict_2', comments='')
#pred_file = open('result.csv', 'w')
#csvfile = csv.writer(pred_file, delimiter=',', lineterminator='\n')
#csvfile.writerow(['id', 'predict_0 - predict_1 - predict_2'])
#csvfile.writerows(zip(ids, preds))

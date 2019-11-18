# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:06:30 2019

@author: 刘万里
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def get_mean(pd_set):
    mean_list = list()
    for i in range(0,13):
        mean_list.append(np.mean(np.array(pd_set[pd_set[i] != '?'][i], dtype = int)))
    for j in range(len(mean_list)):
        mean_list[j] = int(mean_list[j]+0.5)
    return mean_list
    

def deal_data(dataset, mean_list):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if dataset[i][j] == '?':
                dataset[i][j] = mean_list[j]
            else:
                dataset[i][j] = int(dataset[i][j])
    return dataset


df = pd.read_csv('train.csv', names = list(range(14)))
df_test = pd.read_csv('test.csv', names = list(range(13)))
mean_list = get_mean(df)
t_mean_list = get_mean(df_test)
df = deal_data(np.array(df), mean_list)
df_test = deal_data(np.array(df_test), t_mean_list)


#3.使用决策树对测试数据进行类别预测
dtc = DecisionTreeClassifier(criterion = 'gini')
#模型训练
dtc.fit(df[:1700,:13], df[:1700,-1].astype('int'))
#验证集
y_predict = dtc.predict(df[1700:,:13])
#测试集
y_predict1 = dtc.predict(df_test[:,:])

df_p = pd.DataFrame({'id':np.array(range(len(y_predict1)+1))[1:],'y':y_predict1})
df_p.to_csv("./sample-test1.csv",index=False)

true_set = df[1700:,-1]
num = 0
for i in range(len(y_predict)):
    if y_predict[i] == true_set[i]:
        num+=1
        
#验证集的正确率
print(num/len(df[1700:,:]))


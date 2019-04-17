# -*- coding:utf-8 -*-
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import  pandas as pd
import  seaborn as sns
#加载数据
data = pd.read_csv('./breast_cancer_data-master/data.csv')
#数据探索
pd.set_option('display.max_columns', None)
# print(data.head(5))
# print(data.describe())
# print(data.columns)
# print(len(data.columns))
#print(data.shape)
#(569, 32)
#数据可视化,用条形图显示诊断为肿瘤的分布情况
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
sns.countplot(data['diagnosis'])
#plt.show()
'''
数据清洗,注意删除一列后，其他列的下标会发生变化
'''
#由探索发现，共32列，ID列删除；diagnosis转为0/1；剩余30列特征值是10个属性在mean, se, worst三方面的显示，将特征字段分为三段
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])

data.drop('id',axis=1, inplace=True)


data_corr = data[features_mean].corr()   #协方差处理数据类型是series
fig = plt.figure(figsize=(20,20))
sns.heatmap(data_corr,annot=True)
#plt.show()
'''
热力图中对角线上的为单变量自身的相关系数是1。颜色越浅代表相关性越大
则从图中看出第一类：radius_mean ， perimeter_mean， area_mean关联性大
第二类compactness_mean， concavity_mean， concave points_mean关联性大
mean se worst是对同一种类别的不同方面展示，本次选用所有的特征值作为训练集（除id之外）
'''
target = data['diagnosis'].values
print(target.shape)
#print(len(target)) 569
columns = data.columns.tolist()
columns.remove('diagnosis')
#print(len(columns)) 30
features = data[columns]
print(features.shape)
#划分训练集合测试集
train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify=target, random_state=1)
#对数据进行规范化，使每个维度的特征数据均值为0，方差为1
s = StandardScaler()
train_X = s.fit_transform(train_X)
test_X = s.fit_transform(test_X)
#创建SVM分类器，本题使用线性分类器
model = svm.LinearSVC()
#用训练集做训练
model.fit(train_X, train_y)
#用测试集做测试
predict_y = model.predict(test_X)
print("准确率：", accuracy_score(test_y, predict_y))
'''
用线性SVM
target (569,)
train (569, 30)
准确率： 0.9473684210526315
'''
model2 = svm.SVC()
model2.fit(train_X, train_y)
predict2_y = model2.predict(test_X)
print("准确率:", accuracy_score(test_y, predict2_y))
'''
(569,)
(569, 30)
准确率： 0.9473684210526315
准确率: 0.9532163742690059
'''
#综合来看，在这个实战中，相同的训练集合测试集，SVC的准确率高于linearSVC

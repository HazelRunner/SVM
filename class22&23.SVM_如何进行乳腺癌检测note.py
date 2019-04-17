# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics

#加载数据
data = pd.read_csv('./breast_cancer_data-master/data.csv')
#1.数据探索
#由于列较多，把全部列显示做下列配置；若不做配置，显示时则...省略
#pd.set_option('display.max_columns', None)
#print(data.columns)
#32列，出去ID和diagnosis，其余30列属性是：10个特征在3个维度的值
# print(len(data.columns))
#print(data.head(5))
# print(data.describe())
# print(data['diagnosis'].value_counts())
'''
B    357  良性
M    212  恶性
'''

#2.数据清洗
#将特征字段分为3组
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])
#ID列无用，删除
data.drop('id', axis=1, inplace=True)  #用inplace代表修改了原试图的数据
#将B良性替换为0， M恶性替换为1
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0}) #map的用法

#3.数据可视化
#将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'])
plt.show()
#用热力图呈现features_mean 字段间的相关性
corr = data[features_mean].corr()   #计算相关系数，画热力图
print(corr.head(5))
plt.figure(figsize=(14,14))
#annot = true显示每个方格的数据
sns.heatmap(corr, annot=True)
#plt.show()
'''
热力图中对角线上的为单变量自身的相关系数是1。颜色越浅代表相关性越大
则从图中看出第一类：radius_mean ， perimeter_mean， area_mean关联性大
第二类compactness_mean， concavity_mean， concave points_mean关联性大
mean se worst是对同一种类别的不同方面展示，故选一个代表的 mean，现在数据变成10列
从第一类和第二类中各挑取1个作为特征值，如radius_mean ,compactness_mean
现在数据变成6列
'''
#特征选择
features_remain = ['radius_mean', 'texture_mean',
     'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean']
#抽取30%的数据作为测试集，其余作为训练集
train, test = train_test_split(data, test_size=0.3)
#抽取特征选择的数据作为训练和测试数据
train_X = train[features_remain]
test_X =  test[features_remain]
train_y = train['diagnosis']
test_y = test['diagnosis']
#对数据进行规范化，使每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.fit_transform(test_X)

#创建SVM分类器，非线性分类器，默认的是高斯核函数，kernel='rbf'
model = svm.SVC()
#用训练集做训练
model.fit(train_X, train_y)
#用测试集做预测
predict_y = model.predict(test_X)
print("准确率：", metrics.accuracy_score(test_y, predict_y))
#准确率： 0.9239766081871345
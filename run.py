# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:55:33 2021

@author: ankku
""" 
#Mô tả tổng quan về tập dữ liệu
# Bộ dữ liệu chứa các giao dịch được thực hiện bằng thẻ tín dụng vào tháng 9 năm 2013 bởi các chủ thẻ châu Âu. Tập dữ liệu này biểu diễn các giao dịch xảy ra trong hai ngày, trong đó chúng tôi có 492 gian lận trong tổng số 284.807 giao dịch. Tập dữ liệu rất mất cân bằng, lớp positive (gian lận) chiếm 0,172% của tất cả các giao dịch.
#Nó chỉ chứa các biến đầu vào số là kết quả của một phép biến đổi PCA. Rất tiếc, do các vấn đề bảo mật, bộ dữ liệu không thể cung cấp các thuộc tính gốc và thông tin cơ bản về dữ liệu. Các thuộc tính V1, V2, ... V28 là các thành phần chính có được với PCA, các thuộc tính duy nhất không được chuyển đổi với PCA là 'Time' và 'Amount'. Thuộc tính 'Time' chứa số giây trôi qua giữa mỗi giao dịch và giao dịch đầu tiên trong tập dữ liệu. Thuộc tính 'Amount' là số tiền giao dịch, thuộc tính này có thể được sử dụng để học dựa trên chi phí phụ thuộc vào mẫu. Thuộc tính 'Class' là biến phản hồi và nó nhận giá trị 1 trong trường hợp gian lận và 0 nếu không.
#Cảm hứng Xác định các giao dịch gian lận thẻ tín dụng.
#Với tỷ lệ mất cân bằng lớp, chúng tôi đang đo độ chính xác bằng cách sử dụng Khu vực dưới đường cong thu hồi độ chính xác (AUPRC). Độ chính xác của ma trận confusion không có ý nghĩa đối với phân loại không cân bằng.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df= pd.read_csv('creditcard.csv')


#Null Value Check
df.isnull().values.any()


#Data Class Balance Check
print('Fraud Percentage: {}'.format(round((df['Class'].value_counts()[1]/len(df))*100,2)))
print('Non Fraud Percentage: {}'.format(round((df['Class'].value_counts()[0]/len(df))*100,2)))

#Ta vẽ hình minh họa để thấy được sự mất cân bằng trong tập dữ liệu
count= df['Class'].value_counts()
count.plot(kind='bar')
plt.xticks(range(2),['Non Fraud','Fraud'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

#Nhận xét: Bộ dữ liệu ban đầu của chúng ta mất cân bằng. Hầu hết các giao dịch là không gian lận(No Fraud). Nếu chúng ta sử dataframe này làm cơ sở cho các mô hình dự đoán và phân tích, chúng ta có thể gặp rất nhiều lỗi và các thuật toán của chúng tôi có thể sẽ bị thừa vì nó sẽ "giả định" rằng hầu hết các giao dịch không phải là gian lận. Nhưng chúng ta không muốn mô hình của mình bị giả định, chúng ta muốn mô hình của mình phát hiện ra các mẫu có dấu hiệu gian lận!


#-----------PHÂN PHỐI
#Theo phân phối, chúng ta có thể thấy "Amount" giao dịch là rất nhỏ. Còn "Time" của phân bố dàn trải hơn.

fig, ax= plt.subplots(2,1, figsize=(20,10))

amount= df['Amount'].values
time= df['Time'].values

sns.distplot(amount,ax=ax[0], color='r')
sns.distplot(time,ax=ax[1],color='b')
#Nhận xét: Theo phân phối, chúng ta có thể thấy "Amount" giao dịch là rất nhỏ. Còn "Time" của phân bố dàn trải hơn.

#-----------CHUẨN HÓA
from sklearn.preprocessing import RobustScaler # it is prone to outliers
ss1= RobustScaler()
df['Amount']= ss1.fit_transform(df['Amount'].values.reshape(-1, 1))

ss2= RobustScaler()
df['Time']= ss2.fit_transform(df['Time'].values.reshape(-1, 1))

df.head()


#-----------PHÂN TÁCH TẬP DỮ LIỆU(Original DataFrame)
#Trước khi tiếp tục với kỹ thuật Random UnderSampling, chúng ta phải tách dataframe dữ liệu ban đầu. Tại sao? cho mục đích thử nghiệm, mặc dù chúng ta đang tách dữ liệu khi triển khai kỹ thuật Random UnderSampling, hoặc OverSampling , nhưng chúng ta muốn kiểm tra mô hình của mình trên bộ thử nghiệm gốc chứ không phải trên bộ thử nghiệm được tạo bởi một trong hai kỹ thuật này. Mục tiêu chính là làm cho mô hình phù hợp với dataframe được lấy UnderSampling hoặc OverSampling (để các mô hình của chúng ta phát hiện ra các mẫu) và kiểm tra nó trên bộ thử nghiệm ban đầu. 

xorg=df.drop('Class',axis=1)
yorg= df.loc[:,'Class']


from sklearn.model_selection import train_test_split
xorgtrain,xorgtest,yorgtrain,yorgtest= train_test_split(xorg,yorg,test_size=0.2,random_state=9)

print(xorgtrain.shape,xorgtest.shape,yorgtrain.shape,yorgtest.shape)

#-----------RANDOM SAMPLING
#Chúng ta sẽ triển khai "Random Sampling", về cơ bản bao gồm việc xóa dữ liệu để có tập dữ liệu cân bằng hơn và do đó tránh mô hình của chúng ta bị overfitting.

#Tuy nhiên có vấn đề chính với "Random Sampling" là chúng tôi gặp rủi ro rằng các mô hình phân loại của chúng tôi sẽ không hoạt động chính xác như chúng tôi mong muốn vì có rất nhiều mất mát thông tin (492 giao dịch không gian lận từ 284.315 giao dịch không gian lận)



df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492] #Taking top 492 row for 0

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
new_df.shape
df.columns

#-----------CORRELATION MATRICES
#Ma trận tương quan là bản chất của việc hiểu dữ liệu của chúng ta. Chúng tôi muốn biết liệu có những thuộc tính nào ảnh hưởng nhiều đến việc liệu một giao dịch cụ thể có phải là gian lận hay không. Tuy nhiên, điều quan trọng là chúng tôi phải sử dụng dataframe chính xác (subsampling) để chúng ta xem các tính năng nào có mối tương quan positive (Fraud) hoặc negative(Non Fraud) liên quan đến các giao dịch gian lận.

#for Original Data frame
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm_r')

#For new sampled df
#for Original Data frame
plt.figure(figsize=(20,20))
sns.heatmap(new_df.corr(),annot=True,cmap='coolwarm_r')


#Tương quan Negative: V17, V16, V14, V12 và V10 có tương quan nghịch. Lưu ý rằng các giá trị này càng thấp thì càng có nhiều khả năng là một giao dịch gian lận.

#Tương quan Positive: V2, V4, V11 và V19 có tương quan thuận. Lưu ý rằng các giá trị này càng cao thì càng có nhiều khả năng là một giao dịch gian lận.

#BoxPlots: Chúng ta sẽ sử dụng boxplots để hiểu rõ hơn về sự phân bổ của các thuộc tính này trong các giao dịch để biết thấy được có hay không nguy cơ gian gian lận trong giao dịch.

#--------Tương quan Negative
f, axes = plt.subplots(ncols=5, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=new_df, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V16", data=new_df, ax=axes[1])
axes[1].set_title('V16 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, ax=axes[2])
axes[2].set_title('V14 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V12", data=new_df, ax=axes[3])
axes[3].set_title('V12 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V10", data=new_df, ax=axes[4])
axes[4].set_title('V10 vs Class Negative Correlation')

plt.show()

#--------Tương quan Positive
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Postive Correlations with our Class (The higher our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V2", data=new_df, ax=axes[0])
axes[0].set_title('V2 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=new_df, ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V11", data=new_df, ax=axes[2])
axes[2].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V19", data=new_df, ax=axes[3])
axes[3].set_title('V2 vs Class Positive Correlation')

plt.show()


# Boxplots là một cách tiêu chuẩn hóa để hiển thị phân phối dữ liệu dựa trên tóm tắt năm số (“minimun”, first quartile (Q1), median, third quartile (Q3) và “maximum”).

# median (Q2/50th Percentile): giá trị giữa của tập dữ liệu.

# first quartile (Q1/25th Percentile): số chính giữa giữa số nhỏ nhất (không phải số "tối thiểu") và số trung bình của tập dữ liệu.

# third quartile (Q3/75th Percentile): giá trị giữa giữa giá trị trung bình và giá trị cao nhất (không phải "tối đa") của tập dữ liệu.

# interquartile range (IQR): phân vị thứ 25 đến 75.
# Ngoài ra các whiskers (biểu thị bằng màu blue)

# ngoại lệ (biểu thị bằng các vòng tròn green)

# 1. Maximum": Q3 + 1.5 * IQR

# 2. Minimum": Q1 -1,5 * IQR

# 3. outliers = 3 * IQR trở lên


#-------- Phát hiện bất thường
#Visual các phân phối: Trước tiên, chúng ta bắt đầu bằng cách visual sự phân bố của thuộc tính mà chúng ta sẽ sử dụng để loại bỏ một số ngoại lệ. V14 là tính năng duy nhất có phân bố Gaussian so với các thuộc tính V12 và V10.
#Xác định ngưỡng: Sau khi chúng ta quyết định một ngưỡng, chúng ta sẽ sử dụng nó để nhân với iqr (Giá trị thấp hơn outliers sẽ bị loại bỏ), chúng ta sẽ tiến hành xác định ngưỡng trên và ngưỡng dưới bằng cách tính q25 - ngưỡng (ngưỡng cực hạn dưới) và thêm q75 + ngưỡng (ngưỡng cực trên).
#Loại bỏ có điều kiện: Cuối cùng, chúng ta tạo ra một loại bỏ có điều kiện nói rằng nếu "ngưỡng" bị vượt quá ở cả hai thái cực, các phiên bản sẽ bị loại bỏ.
#Biểu diễn Boxplot: Visual thông qua boxplot rằng số lượng "extreme outliers" đã được giảm xuống một lượng đáng kể.

from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)


v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()
#-------- Xử lý ngoại lệ
#Như chúng ta nói trong biểu đồ boxplot, tất cả các biến có mối tương quan đều có giá trị ngoại lệ, vì vậy bây giờ chúng ta sẽ xử lý chúng bằng iqr, lb và ub.
df2= new_df # I am creating the copy of new_df to preserve the original data
treat= ['V14','V12','V10']
for j in treat:
    q25,q75= new_df[j].quantile(q=0.25),new_df[j].quantile(q=0.75)
    iqr= q75-q25
    cut_off= iqr*1.5
    lb,ub= q25-cut_off,q75+cut_off
    outliers= [x for x in new_df[j] if x<=lb or x>=ub]
    print(j,'Q25: {} , Q75: {}, IQR: {}, Cutoff: {}, LB: {}, UB: {},'.format(q25,q75,iqr,cut_off,lb,ub))
    print(len(outliers), outliers)
    df2= df2.drop(df2[(df2['V14'] > ub) | (df2['V14']< lb)].index, axis=0)
    print(df2.shape)
    print('----' * 44)

#Kiểm tra kết quả sau khi xử lý ngoại lệ
df2.shape

from collections import Counter
Counter(df2['Class'])

f, axes = plt.subplots(ncols=len(treat), figsize=(20,4))
for i,j in enumerate(treat):
# Postive Correlations with our Class (The higher our feature value the more likely it will be a fraud transaction)
    sns.boxplot(x="Class", y=j, data=df2, ax=axes[i])
    axes[i].set_title(j)

from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

x=df2.drop('Class',axis=1).values
y= df2.loc[:,'Class'].values

# SPlitting the test and train after removing outliers

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

#Ta thực hiện fitting với các models và tính toán toán traning và testing_score
classifier= {
    'Logistic Regression':LogisticRegression(),
    'KNN':KNeighborsClassifier(),
    'SVC':SVC(),
    'DecisionTree':DecisionTreeClassifier()
}

import warnings
warnings.filterwarnings('ignore')
#Thực hiện tính toán trên dữ liệu gốc
for key,values in classifier.items():
    values.fit(xtrain,ytrain)
    training_score= cross_val_score(values,xtrain,ytrain,cv=5)
    print('Training accuracy score of {} is {}'.format(key,round(training_score.mean()*100,2)))
    train_pred = cross_val_predict(values, xtrain, ytrain, cv=5)
    print('Roc_Auc training score for {} is {}: '.format(key, round(roc_auc_score(ytrain,train_pred)*100,2)))
    test_score= cross_val_score(values,xtest,ytest,cv=5)
    print('Test accuracy score of {} is {}'.format(key,round(test_score.mean()*100,2)))
    test_pred = cross_val_predict(values, xtest, ytest, cv=5)
    print('Roc_Auc test score for {} is {}: '.format(key, round(roc_auc_score(ytest,test_pred)*100,2)))
    print('---'*30)

#Dựa vào kết quả thu được sau khi tính toán trên dữ liệu gốc, chúng ta thấy accuracy chưa thực sự cao. Do vậy ta cần điều chỉnh tham số để thu được kết quả phù hợp hơn.


#--------Điều chỉnh Hyper Parameter sử dụng GridSearchCv

from sklearn.model_selection import GridSearchCV
# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}


classifier= {
    'Logistic Regression':LogisticRegression(),
    'KNN':KNeighborsClassifier(),
    'SVC':SVC(),
    'DecisionTree':DecisionTreeClassifier()
}

def grid_search(classifier,Param):
    grid_log_reg = GridSearchCV(classifier,param_grid=Param)
    grid_log_reg.fit(xtrain, ytrain)
    best_param = grid_log_reg.best_estimator_
    print('{} algorithm best parameter are : {}'.format(classifier.__class__.__name__,best_param))


grid_search(LogisticRegression(),log_reg_params)

log_reg= LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=None, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)

log_reg.fit(xtrain,ytrain)

log_reg_pred = cross_val_predict(log_reg, xtest, ytest, cv=5,method="decision_function")

from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold= roc_curve(ytest,log_reg_pred)

plt.figure(figsize=(15,10))
plt.plot([0,1],[0,1],'r--')
plt.plot(fpr,tpr,'g')
plt.title('Auc Score is :'+str(auc(fpr,tpr)))


##--------TÌM HIỂU SÂU HƠN VỀ BỘ PHÂN LỚP HỒI QUY LOGISTIC












#--------Kiểm tra mạng nơ-ron đối với Under và Oversample data
import keras
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

#Chúng ta tạo mạng nơ ron 2 lớp ẩn đơn giản và sẽ sử dụng nó để phân lớp
classifier= Sequential()
classifier.add(Dense(15, activation='relu',kernel_initializer='uniform',input_shape=(30,)))
classifier.add(Dense(15, activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform' ))

classifier.summary()


classifier.compile(optimizer='adam',loss= ['binary_crossentropy'],metrics=['accuracy'])
X = df.drop('Class', axis=1)
y = df['Class']
from imblearn.under_sampling import RandomUnderSampler  
X_nearmiss, y_nearmiss = RandomUnderSampler().fit_resample(X.values, y.values) 

classifier.fit(X_nearmiss,y_nearmiss, batch_size=10,epochs=100)

undersample_pred= classifier.predict_classes(xorgtest,batch_size=200,verbose=0)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
labels = ['No Fraud', 'Fraud']
confusion_matrix(yorgtest,undersample_pred)
print(classification_report(yorgtest, undersample_pred, target_names=labels))


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
precision, recall, threshold = precision_recall_curve(yorgtest, undersample_pred)

plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          average_precision_score(yorgtest, undersample_pred)), fontsize=16)


# SMOTE Technique (OverSampling) After splitting and Cross Validating
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


# This will be the data were we are going to 
Xsm_train, ysm_train = sm.fit_resample(X, y)

classifier.fit(Xsm_train,ysm_train, batch_size=200,epochs=100)


oversample_pred= classifier.predict_classes(xorgtest,batch_size=200,verbose=0)
confusion_matrix(yorgtest,oversample_pred)


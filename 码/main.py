import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor as RF


#read data
print(os.getcwd())
data_ori = pd.read_csv('temperature_prediction.csv')
data=data_ori
#change category in a more reasonalble name
data=np.array(data)
for t in range(1,5):    
    data[1550*t:1550*(t+1),1]=data[0:1550,1]

#baseline trival
#x_trival=x.drop([1],axis=1)
    
# #drop data which are nan, number is 164
data=pd.DataFrame(data)
data =data.dropna(axis=0)
label=data[23] 
label=np.array(label)  
data=data.drop([23,24],axis=1)
data=np.array(data)

#data split for preprocess
x, x_un, y, y_un=train_test_split(data, label, test_size=0.1, \
    train_size=0.9, random_state=50)
x_tr, x_te, y_tr, y_te=train_test_split(x, y, test_size=0.22, \
    train_size=0.78, random_state=50)
x_tr, x_val, y_tr, y_val=train_test_split(x_tr, y_tr, test_size=0.2, \
    train_size=0.8, random_state=50)

#if use 1:61number to substitute
t=1
date=[]
for i in range (0,len(data_ori)-2):
    if i%25==0 and i>=25:
        t=t+1
    if i%1550==0:
            t=1
    date.append(t)
data_num=np.array(data_ori)
date=np.array(date)
data_num[0:len(date),1]=date
data_num=pd.DataFrame(data_num)
data_num =data_num.dropna(axis=0)
data_num=data_num.drop([23,24],axis=1)
data_num=np.array(data_num)
x, x_un, y, y_un=train_test_split(data_num, label, test_size=0.1, \
    train_size=0.9, random_state=50)
x_tr, x_te, y_tr, y_te=train_test_split(x, y, test_size=0.22, \
    train_size=0.78, random_state=50)
x_tr, x_val, y_tr, y_val=train_test_split(x_tr, y_tr, test_size=0.2, \
    train_size=0.8, random_state=50)
   
temp_tr=x_tr
temp_un=x_un
temp_val=x_val      

#from above result choose change category from 1 to 61, now use 1-nn for unknown data
def std_h(x_tr,x_val):
    svm_tr_std = preprocessing.StandardScaler().fit(x_tr)#standardize first
    x_tr=svm_tr_std.transform(x_tr)
    x_val = svm_tr_std.transform(x_val)
    return x_tr,x_val
x_tr,x_val=std_h(x_tr,x_val)
 
# svm predict unknown data  
svm_un_std = preprocessing.StandardScaler().fit(x_un)
x_un=svm_un_std.transform(x_un)

svm_r=svm.SVR().fit(x_tr,y_tr)
svm_y_pre=svm_r.predict(x_un)
x_tr_all=np.insert(x_tr,0,x_un,axis=0)
y_tr_all=np.insert(y_tr,0,svm_y_pre,axis=0)

#combine data
x_tr_lasso1=np.insert(temp_tr,0,temp_un,axis=0)
x_tr_lasso=np.insert(x_tr_lasso1,0,temp_val,axis=0)
x_tr_h=x_tr_lasso
y_tr_lasso=np.insert(y_tr_all,0,y_val,axis=0)
y_tr_h=y_tr_lasso

model_final=RF(n_estimators=89,random_state=10,n_jobs = -1)
x_tr_final=x_tr_h[:,range(6,len(x_tr_h[1,:]))]
x_te_final=x_te[:,range(6,len(x_te[1,:]))]
x_tr_final,x_te_final=std_h(x_tr_h,x_te)
model_final.fit(x_tr_final, y_tr_h)
y_pred_final=model_final.predict(x_te_final)
score_final=mean_squared_error(y_pred_final, y_te)
print('final model MSE for test set',score_final)
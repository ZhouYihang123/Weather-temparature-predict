#In[1]
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import AdaBoostRegressor as AB
from sklearn.metrics import r2_score

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
#trival baseline mse
mse_trival=0
for i in range (0,len(y_val)):
    mse_trival=mse_trival+(y_val[i]-np.mean(y_tr))**2
mse_trival=mse_trival/len(y_val)
print('trival baseline MSE is:')
print(mse_trival)
print(' ')

# non-trival baseline mse
 #if delete category feature without standardize
x_tr_del=np.delete(x_tr,1,1)
x_val_del=np.delete(x_val,1,1)
reg_del_no = linear_model.LinearRegression().fit(x_tr_del, y_tr)
reg_del_pre_no = reg_del_no.predict(x_val_del)
del_mse_no=mean_squared_error(y_val,reg_del_pre_no)
print('Without standardize,the MSE is:' , del_mse_no)

def mse_lin(x_tr_del,x_val_del,y_val,y_tr):
    std = preprocessing.StandardScaler().fit(x_tr_del)#standardize first
    x_tr_del = std.transform(x_tr_del)
    x_val_del = std.transform(x_val_del)
    reg_del = linear_model.LinearRegression().fit(x_tr_del, y_tr)
    reg_del_pre = reg_del.predict(x_val_del)
    del_mse=mean_squared_error(y_val,reg_del_pre )
    return del_mse

print('if delete the category feature, the non-trival baseline MSE of validation is:' \
      ,mse_lin(x_tr_del,x_val_del,y_val,y_tr))
# print('the test set MSE is:',mse_lin(x_tr_del,x_val_del,y_val,y_tr))
# print(' ')


 #if use one hot code change   
def one_hot_trans(x_tr):
    category=x_tr[:,1]
    category=np.mat(category)
    category=np.transpose(category)
    one_hot = OneHotEncoder()
    one_hot.fit(category)
    c_feat=one_hot.transform(category).toarray()
    
     #use TruncatedSVD to reduce dimension,pca no support sparse input
    svd = TruncatedSVD(n_components=3)
    c_svd=svd.fit_transform(c_feat)
    x_tr=np.delete(x_tr,1,axis=1)
    x_tr=np.insert(x_tr,1,c_svd[:,0],axis=1)
    x_tr=np.insert(x_tr,2,c_svd[:,1],axis=1)
    x_tr=np.insert(x_tr,3,c_svd[:,2],axis=1)
    return x_tr
x_tr=one_hot_trans(x_tr)
x_val=one_hot_trans(x_val)
print('if use one-hot code to change category feature, the non-trival baseline MSE is:')
print(mse_lin(x_tr,x_val,y_val,y_tr))
print(' ')

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
print('if change category to the date,the non-trival baseline MSE is: ')
print(mse_lin(x_tr,x_val,y_val,y_tr))
print(' ')      

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

print('if add unknown data,the non-trival baseline MSE is: ')
print(mse_lin(x_tr_all,x_val,y_val,y_tr_all))
print(' ')

# Lasso regression with regularization
x_tr_lasso1=np.insert(temp_tr,0,temp_un,axis=0)
x_tr_lasso=np.insert(x_tr_lasso1,0,temp_val,axis=0)
x_tr_h=x_tr_lasso
y_tr_lasso=np.insert(y_tr_all,0,y_val,axis=0)
y_tr_h=y_tr_lasso

folds = KFold(n_splits=5,shuffle=True,random_state=0)





# In[2]
scores=[]
scores_mean=[]
def cross_val_h(x_tr_lasso,y_tr_lasso,model):
    for i, (train_index, valid_index) in enumerate(folds.split(x_tr_lasso, y_tr_lasso)):
        X_train,X_valid = x_tr_lasso[train_index],x_tr_lasso[valid_index]
        y_train,y_valid = y_tr_lasso[train_index],y_tr_lasso[valid_index]   
        X_train,X_valid=std_h(X_train,X_valid)
        model.fit(X_train,y_train)
        y_pred_valid = model.predict(X_valid)
        scores.append(mean_squared_error(y_valid, y_pred_valid))
        scores_mean=np.mean(scores)
    return scores_mean

def test_acc(x_te_lasso,y_te,model):
    y_pred_test=model.predict(x_te_lasso)
    scores=mean_squared_error(y_pred_test, y_te)
    return scores


def best_para(model,lamda):    
    t=0
    lasso_mean=[[0]*2 for i in range(len(lamda))]
    for i in lamda:
        lasso= eval(model)
        mean_h =cross_val_h(x_tr_lasso,y_tr_lasso,lasso)
        lasso_mean[t][0]=i
        lasso_mean[t][1]=mean_h
        t=t+1
    lasso_mean=np.array(lasso_mean)
    plt.plot(lasso_mean[:,0], lasso_mean[:,1])
    plt.show
    plt.xlabel("log(lambda)")
    plt.ylabel("MSE")
    plt.figure()
    r_l = np.where(lasso_mean[:,1]== np.min(lasso_mean[:,1]))
    r_l=int(r_l[0])
    print('the best parameter in validation set  is:')
    print(10**(lasso_mean[r_l,0]))
    print('The MSE for validation set is:',lasso_mean[r_l,1])
    i=lasso_mean[r_l][0]
    lasso_final=eval(model)
    x_tr_std,x_te_std=std_h(x_tr_lasso,x_te)
    lasso_final.fit(x_tr_std,y_tr_lasso)
    y_pred_test=lasso_final.predict(x_te_std)
    scores=mean_squared_error(y_pred_test, y_te)
    print('The MSE for test set is:',scores)
    return lasso_final

# lasso regression
model='linear_model.Lasso(alpha=10**(i), normalize=False,random_state=5)'
lamda=np.linspace(-4,-1,100)
#lamda=np.linspace(10**(-4),10**(-3),100)
model_lasso=best_para(model,lamda)
print('not important feature is 6,19')
print('')

# In[3]
##ridge regression
model2='linear_model.Ridge(alpha=10**(i), normalize=False,random_state=5)'
lamda2=np.linspace(-4,3,100)
model_ridge=best_para(model2,lamda2)


# In[5]
##random forest model
x_tr_r, x_val_r, y_tr_rf, y_val_rf=train_test_split(x_tr_h, y_tr_h, test_size=0.2, \
    train_size=0.8, random_state=50)
x_tr_rf,x_val_rf=std_h(x_tr_r,x_val_r)
x_tr_rf,x_te_rf=std_h(x_tr_r,x_te)


def rf_ab(model,tree,x_tr_rf,x_val_rf,x_te_rf):
    score_rf=[]
    for i in tree:
        model_rf = eval(model)
        model_rf.fit(x_tr_rf,y_tr_rf)
        y_pred_rf=model_rf.predict(x_val_rf)
        score_rf.append(mean_squared_error(y_pred_rf, y_val_rf))
    r_1 = np.where(score_rf== np.min(score_rf))
    r_1=int(r_1[0])
    plt.plot(tree,score_rf)
    plt.show
    plt.xlabel("number of trees")
    plt.ylabel("MSE")
    plt.figure()   
    print('the best parameter in validation set  is:')
    print(int(tree[r_1]))
    print('The MSE for validation set is:',score_rf[r_1])
    i=tree[r_1]
    model_rf_final = eval(model)
    model_rf_final.fit(x_tr_rf,y_tr_rf)
    y_pred_rf=model_rf.predict(x_val_rf)
    score_r2=r2_score(y_val_rf, y_pred_rf)
    print('r2_score is', score_r2)
    print('The MSE for test set is:',test_acc(x_te_rf,y_te,model_rf_final))
    print(' ')
    return model_rf,score_rf[r_1]

tree=np.linspace(10,300,150)
model4='RF(n_estimators=int(i),random_state=10,n_jobs = -1)'
model_rf,score_rf=rf_ab(model4,tree,x_tr_rf,x_val_rf,x_te_rf)



# In[5]
#adaboost
tree=np.linspace(100,600,450)
print('when learning rate is 3.4')
model5_1='AB(n_estimators=int(i),random_state=10,learning_rate=3.4)'
model5_ab1,score_ab1=rf_ab(model5_1,tree,x_tr_rf,x_val_rf,x_te_rf)
print('when learning rate is 3.5')
model5_2='AB(n_estimators=int(i),random_state=10,learning_rate=3.5)'
model5_ab2,score_ab2=rf_ab(model5_2,tree,x_tr_rf,x_val_rf,x_te_rf)
print('when learning rate is 3.6')
model5_3='AB(n_estimators=int(i),random_state=10,learning_rate=3.6)'
model5_ab3,socre_ab3=rf_ab(model5_3,tree,x_tr_rf,x_val_rf,x_te_rf)


    # In[6]
# random forest delete feature
im=[0.00682536,0.0101776,0.0560541,0.0143519,0.011779,0.0098689,0.70069,\
     0.0172333,0.0215572,0.0141699,0.029013,0.0120514,0.0237812,0.020496,0.00724137, \
         0.00745542,0.00508053,0.00367521,0.00390143,0.00590958,0.00467821, \
             0.00340986,0.0109691]
array = np.array(im)
order = array.argsort()
x_tr_rank=x_tr_rf[:,order]
x_val_rank=x_val_rf[:,order]
x_te_rank=x_te_rf[:,order]
print(order)

tree=np.linspace(10,300,150)
score_all=[]
t=1


for i in range(0,len(x_tr_rank[1,:])-3):
    print('if delete',i+1,'features')
    x_tr_6=x_tr_rank[:,range(i+1,len(x_tr_rank[1,:]))]
    x_val_6=x_val_rank[:,range(i+1,len(x_tr_rank[1,:]))]
    x_te_6=x_te_rank[:,range(i+1,len(x_tr_rank[1,:]))]
    model6='RF(n_estimators=int(i),random_state=10,n_jobs = -1)'
    model_rf,score_del=rf_ab(model6,tree,x_tr_6,x_val_6,x_te_6)
    score_all.append(score_del)
plt.plot(range(1,19,1),score_all[0:18])
plt.show
plt.xlabel("number of fetures delete")
plt.ylabel("MSE")
plt.figure()     

    
model_final=RF(n_estimators=89,random_state=10,n_jobs = -1)
x_tr_final=x_tr_h[:,range(6,len(x_tr_h[1,:]))]
x_te_final=x_te[:,range(6,len(x_te[1,:]))]
x_tr_final,x_te_final=std_h(x_tr_h,x_te)
model_final.fit(x_tr_final, y_tr_h)
y_pred_final=model_final.predict(x_te_final)
score_final=mean_squared_error(y_pred_final, y_te)
print('final model MSE for test set',score_final)

# In[7] plot features relevent

x_tr_rf=x_tr_rank
plt.scatter(x_tr_rf[:,22],y_tr_rf,color='green', label='training data')
x_grid = np.arange(min(x_tr_rf[:,22]), max(x_tr_rf[:,22]), 0.001)
model_p=RF(n_estimators=89,random_state=10,n_jobs = -1)
feature=x_tr_rf[:,22]
feature=feature.reshape(-1,1)
model_p.fit(feature,y_tr_rf)
x_grid=x_grid.reshape(-1,1)
y_pred_val=model_p.predict(x_grid)
plt.plot(x_grid,y_pred_val,color='red',label='regression function')
plt.title('Random Forest Regression')
plt.xlabel('LDAPS_Tmax_lapse')
plt.ylabel('Next_Tmax')
plt.legend(loc='best')
plt.show()

plt.scatter(x_tr_rf[:,21],y_tr_rf,color='green', label='training data')
x_grid = np.arange(min(x_tr_rf[:,21]), max(x_tr_rf[:,21]), 0.001)
model_p=RF(n_estimators=89,random_state=10,n_jobs = -1)
feature=x_tr_rf[:,21]
feature=feature.reshape(-1,1)
model_p.fit(feature,y_tr_rf)
x_grid=x_grid.reshape(-1,1)
y_pred_val=model_p.predict(x_grid)
plt.plot(x_grid,y_pred_val,color='red',label='regression function')
plt.title('Random Forest Regression')
plt.xlabel('Present Tmax')
plt.ylabel('Next_Tmax')
plt.legend(loc='best')
plt.show()

# In[8]i
th=np.arange(min(y_te),max(y_te),0.1)
th_sel=[[0]*2 for i in range(len(y_te))]
t=0
i=np.mean(y_pred_final)
c=0
for j in range(0,len(y_te)):
    if y_pred_final[j]<=i and y_te[j]<=i:
        c=c+1
    if y_pred_final[j]>i and y_te[j]>i:
        c=c+1
err=1-c/len(y_pred_final)
print('transform to 2 classification,error is',err)


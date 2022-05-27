# Clustering Model
#A python file that predicts loan grade for Problem 2 using KNN

#Imports

#Data
import pandas as pd
import numpy as np

#Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Other
import pickle

#General
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import OneHotEncoder

#knn
from sklearn.neighbors import KNeighborsClassifier

#Getting the csv file's name with the wrangled data
data_file='wrang_xyz_data.csv'

#Splitting the data into different categories that make sense
loan_data=['purpose','initial_list_status','term','loan_amnt']
emp_data=['emp_length','collections_12_mths_ex_med','acc_now_delinq',
'home_ownership','annual_inc','verification_status','delinq_2yrs',
'inq_last_6mths','open_acc','pub_rec','total_acc','earliest_cr_line','dti',
'tot_cur_bal','tot_coll_amt'] #address
ohe_cols=['purpose','verification_status','home_ownership','initial_list_status','term'] #address
#The following inputs are left out as they are only useful for problem 1.
#out=['last_pymnt_d','last_credit_pull_d','recoveries','collection_recovery_fee',
# 'last_pymnt_amnt','total_pymnt','total_rec_int','int_rate','out_prncp',
# ''total_rec_late_fee','default_ind']

#Getting the features that will be included in the model
features=loan_data+emp_data

def one_hot_encode(x_train,x_test,ohe_cols=ohe_cols,pickled='no',ohe_name=''):
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(x_train[ohe_cols])

        if pickled=='yes':
                pickle.dump(ohe,open(ohe_name,'wb'))
        else:
                pass
                
        x_train_enc = pd.DataFrame(ohe.transform(x_train[ohe_cols]).toarray(),index=x_train.index)
        x_train=x_train.join(x_train_enc).drop(ohe_cols,axis=1)
        x_train.columns = x_train.columns.map(str)

        x_test_enc = pd.DataFrame(ohe.transform(x_test[ohe_cols]).toarray(),index=x_test.index)
        x_test=x_test.join(x_test_enc).drop(ohe_cols,axis=1)
        x_test.columns = x_test.columns.map(str)

        return x_train,x_test

def scaler(x,ohe_cols=ohe_cols,pickled='no',ohe_name=''):
        X_scale=x.drop(ohe_cols,axis=1)
        scaler = StandardScaler()
        scaler.fit(X_scale)

        if pickled=='yes':
                pickle.dump(scaler,open(ohe_name,'wb'))
        else:
                pass
                
        X_scale = pd.DataFrame(scaler.transform(X_scale),index=x.index,columns=X_scale.columns)
        x=X_scale.join(x[ohe_cols])
        return x

#Clustering model predicting loan grade for problem 2
#The model's inputs are the data_file which should be set equal to the
#  wrangled data file and the value to be predicted 
#(pred_value) which in this case is the 'grade'. It could also be the 'sub_grade'

def get_model_clus(data_file='wrang_xyz_data.csv',pred_value=['grade']):
    #Imporitng the wrangled csv file and including the useful columns for it
    df = pd.read_csv('data/'+data_file,usecols=features+pred_value)

    #Creating X and y variables for input and output
    X=df.drop(pred_value[0],axis=1)
    y=df[pred_value[0]]

    X_scale=X.drop(['purpose','verification_status','home_ownership',
    'initial_list_status','term'],axis=1)
    X_non_scale=X[['purpose','verification_status','home_ownership',
    'initial_list_status','term']]

    #Scaling the data
    X=scaler(X,ohe_cols=ohe_cols,pickled='no',ohe_name='scaler_clust')

    #Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
    train_size=0.75, test_size=0.25)

    #One-hot Encoding
    X_train,X_test=one_hot_encode(X_train,X_test,pickled='no',ohe_name='ohe_clust')

    #Using the K-Neirghest-Neighours algorith for clustering
    model_clus = KNeighborsClassifier(n_neighbors=5)
    model_clus.fit(X_train,y_train)
    y_clus_pred = model_clus.predict(X_test)

    #Printing useful metrics
    print('Clustering model predicting grade')
    print(confusion_matrix(y_test, y_clus_pred))
    print(classification_report(y_test, y_clus_pred))

    return model_clus
#Running the function 
model_clus=get_model_clus(pred_value=['grade'])


#This code will print a graph of error vs k value which will
#  let you choose an appropriate 
#K
def get_k_value(data_file='wrang_xyz_data.csv',pred_value=['grade']):

    features=['int_rate']
    df = pd.read_csv('data/'+data_file,usecols=features+pred_value)

    #Creating X and y variables for input and output
    X=df.drop(pred_value[0],axis=1)
    y=df[pred_value[0]]

    #Scaling the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    #Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
     train_size=0.75, test_size=0.25)

    #One-hot Encoding
    #ohe = OneHotEncoder(handle_unknown='ignore')
    #ohe.fit(X_train)
    #X_train = ohe.transform(X_train)
    #X_test = ohe.transform(X_test)

    #Creating empty error rate list
    error_rate = []

    #Running the algorithm for different values of k
    for i in range(1,10):
        
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed',
     marker='o',
        markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')

#Running the clustering model for sub_grade, but with intrest rate
#  being the only input as it is highly correlated to grade and sub-grade
def get_model_clus(data_file='wrang_xyz_data.csv',pred_value=['grade']):
    #Imporitng the wrangled csv file and including the useful columns for it
    features=['int_rate']
    df = pd.read_csv('data/'+data_file,usecols=features+pred_value)

    #Creating X and y variables for input and output
    X=df.drop(pred_value[0],axis=1)
    y=df[pred_value[0]]

    #Scaling the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    #Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
     train_size=0.75, test_size=0.25)

    #Using the K-Neirghest-Neighours algorith for clustering
    model_clus = KNeighborsClassifier(n_neighbors=7)
    model_clus.fit(X_train,y_train)
    y_clus_pred = model_clus.predict(X_test)

    #Printing useful metrics
    print('int_rate clustering model predicting sub_grade')
    print(confusion_matrix(y_test, y_clus_pred))
    print(classification_report(y_test, y_clus_pred))

    return model_clus
    
#Running the function 
model_clus=get_model_clus(pred_value=['sub_grade'])
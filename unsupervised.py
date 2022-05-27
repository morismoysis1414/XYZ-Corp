# Unsupervised Clustering Model
#A python file that predicts loan grade for Problem 2 using unsupervised 
#learning and K-means square
#Imports

#Data
import pandas as pd

#Other
import pickle

#General
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#knn
from sklearn.cluster import KMeans

#Getting the csv file's name with the wrangled data
data_file='wrang_xyz_data.csv'

#Splitting the data into different categories that make sense
loan_data=['initial_list_status','term','loan_amnt'] #purpose
emp_data=['emp_length','collections_12_mths_ex_med','acc_now_delinq',
'annual_inc','verification_status','delinq_2yrs','inq_last_6mths',
'open_acc','pub_rec','total_acc','earliest_cr_line','dti',
'tot_cur_bal','tot_coll_amt'] #address,home_ownership
ohe_cols=['purpose','verification_status','home_ownership','initial_list_status','term'] #address
#The following inputs are left out as they are only useful for problem 1.
#out=['last_pymnt_d','last_credit_pull_d','recoveries',
# 'collection_recovery_fee','last_pymnt_amnt','total_pymnt','total_rec_int',
# 'int_rate','out_prncp',''total_rec_late_fee','default_ind']

#Getting the features that will be included in the model
features=loan_data+emp_data

def one_hot_encode(x,ohe_cols=ohe_cols,pickled='no',ohe_name=''):
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(x[ohe_cols])

        if pickled=='yes':
                pickle.dump(ohe,open(ohe_name,'wb'))
        else:
                pass
                
        x_enc = pd.DataFrame(ohe.transform(x[ohe_cols]).toarray(),
        index=x.index)
        x=x.join(x_enc).drop(ohe_cols,axis=1)
        x.columns = x.columns.map(str)

        return x

def scaler(x,ohe_cols=ohe_cols,pickled='no',ohe_name=''):
        X_scale=x.drop(ohe_cols,axis=1)
        scaler = StandardScaler()
        scaler.fit(X_scale)
        
        if pickled=='yes':
                pickle.dump(scaler,open(ohe_name,'wb'))
        else:
                pass
                
        X_scale = pd.DataFrame(scaler.transform(X_scale),
        index=x.index,columns=X_scale.columns)
        x=X_scale.join(x[ohe_cols])
        return x

#Clustering unsupervised model predicting loan grade for problem 2
#The model's inputs are the data_file which should be set equal to the
#wrangled data file and the value to be predicted 
#(pred_value) which in this case is the 'grade'. It could also be the 'sub_grade'

def get_model_unsup(data_file='wrang_xyz_data.csv',pred_value=['grade']):
    #Imporitng the wrangled csv file and including the useful columns for it
  
    df = pd.read_csv('data/'+data_file,usecols=features+pred_value) #int_rate

    X=df.drop('grade',axis=1)

    #Scaling    
    X=scaler(X,ohe_cols=ohe_cols,pickled='no',ohe_name='scaler_unsup')

    #One-hot Encoding
    X=one_hot_encode(X,pickled='no',ohe_name='ohe_unsup')

    #Using the K-Means-Square algorith for clustering
    model_unsup = KMeans(n_clusters=7)
    model_unsup.fit(X)

    #Creating a function assigning a grade to a label
    def converter(cluster):
        clust={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
        return clust[cluster]
        
    df['Cluster'] = df['grade'].apply(converter)

    #Printing useful metrics
    print(confusion_matrix(df['Cluster'],model_unsup.labels_))
    print(classification_report(df['Cluster'],model_unsup.labels_))

    return model_unsup
#Running the function 
model_unsup=get_model_unsup(pred_value=['grade'])
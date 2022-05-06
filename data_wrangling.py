# Data Wrangling
#A python file that wrangles the data and creates 'wrang_xyz_data.csv'.

#Imports

#Data
import pandas as pd
import numpy as np

#Date
import datetime as dt
from datetime import datetime

#Function that wrangles the data
def wrangle_data(file='data/xyz_corp_lending_data.csv'):
    
    #Read CSV file
    df=pd.read_csv('data/xyz_corp_lending_data.csv',sep='\t')

    #Drop columns (to be explained why for each column)
    df=df.drop(['policy_code','funded_amnt','funded_amnt_inv','out_prncp_inv',
    'total_pymnt_inv','id','member_id','total_rec_prncp','policy_code',
    'installment','pymnt_plan','application_type','next_pymnt_d','title',
    'emp_title'],axis=1)

    #Creating a metric to get % of null values and selecting the columns
    #  that have less than 50%
    nullity=(df.isnull().sum()/len(df)*100).sort_values(ascending=False)
    df=df[nullity[nullity<50].index]

    #Removing text from emp_length and term columns and converting to float
    df['emp_length']=df['emp_length'].str.replace('[A-Za-z\s+<>]+',
     '').astype('float')
    df['term']=df['term'].str.replace('[A-Za-z\s+<>]+', '').astype('float')

    #Converting Date Columns to YYYYMMDD format and float
    date_cols=['last_pymnt_d','last_credit_pull_d','issue_d',
    'earliest_cr_line']
    for col in date_cols:
        df[col]=pd.to_datetime(df[col]).dt.strftime('%Y%m%d').astype(float)

    #Filling na with median or mode
    med=['tot_cur_bal','tot_coll_amt','total_rev_hi_lim','revol_util']
    mod=['emp_length','last_pymnt_d','collections_12_mths_ex_med',
    'last_credit_pull_d']
    for col in med:
        df[col]=df[col].fillna(df[col].median())

    for col in mod:
        df[col]=df[col].fillna(df[col].mode()[0])

    #Cleaning up zip_code to only have numbers
    df['zip_code']=df['zip_code'].str.replace('[A-Za-z\s+<>]+',
     '')#.astype('float')
    df['address']=df['zip_code']+df['addr_state']
    df=df.drop(['zip_code','addr_state'],axis=1)

    #Combning two rows that are the same in verification_status column
    df['verification_status']=df['verification_status'].str.replace(
        'Source Verified','Verified')

    #Encoding categorical values 
    #cat_columns=['purpose','verification_status','home_ownership',
    # 'initial_list_status','address','term']
    #df[cat_columns] = df[cat_columns].astype('category').
    # apply(lambda x: x.cat.codes)

    #Cleaning outliers
    df=df[(df['dti']<150)&(df['revol_util']<200)&(df['acc_now_delinq']<13)&(
        df['tot_coll_amt']<1000000)&(df['tot_cur_bal']<4000000)&(
            df['total_rev_hi_lim']<2000000)&(df['out_prncp']<35001)]
    return df

#Running the function 
wrangle_data()#.to_csv('data/wrang_xyz_data.csv',index=False)
#mports
from flask import Flask, request
import logging
import pickle
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler



logging.basicConfig(filename='dump.log', level=logging.INFO)

app = Flask(__name__)

@app.route("/predict", methods=["GET"])


def predict_loan():
    
    #Getting data from JSON file
    req_j=request.json
    data=req_j

    ohe_cols=['purpose','verification_status','home_ownership','initial_list_status','term'] #address

    #Creating dataframe
    df=pd.DataFrame.from_records([data])
    df_nlp=df['desc']
    df=df.drop('desc',axis=1)
    
    def one_hot_encode(ohe_name,df=df,ohe_cols=ohe_cols):
        ohe_model=pickle.load(open(ohe_name, 'rb'))
        df_enc = pd.DataFrame(ohe_model.transform(df[ohe_cols]).toarray(),
        index=df.index)
        df_model=df.join(df_enc).drop(ohe_cols,axis=1)
        df_model.columns = df_model.columns.map(str)
        return df_model

    def scaler(scaler_name,df=df,ohe_cols=ohe_cols):
            df_scale=df.drop(ohe_cols,axis=1)
            scaler_model=pickle.load(open(scaler_name, 'rb'))
            df_scaled = pd.DataFrame(scaler_model.transform(df_scale),index=df.index,columns=df_scale.columns)
            df_unsup=df_scaled.join(df[ohe_cols])
            return df_unsup

    #Unpickling nlp model predicting purpose
    model_nlp = pickle.load(open('model_nlp', 'rb'))
    vect_nlp=pickle.load(open('vect_nlp', 'rb'))
    df_nlp=vect_nlp.transform(df_nlp)
    prediction_nlp=model_nlp.predict(df_nlp)[0]
    df['purpose']=prediction_nlp

    clust={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'}


    #Unpickling classification model predicting default index
    model_class_def = pickle.load(open('model_class_def', 'rb'))
    df_class_def=one_hot_encode('ohe_class_def')
    prediction_class_def=model_class_def.predict(df_class_def)
    #Unpickling regression model predicting interest rate
    model_reg_int=pickle.load(open('model_reg_int', 'rb'))
    df_reg_int=one_hot_encode('ohe_reg_int')
    prediction_reg_int=model_reg_int.predict(df_reg_int)
    intr_rate=prediction_reg_int[0]

    #Unpickling unsupervised clustering model predicting grade
    model_unsup=pickle.load(open('model_unsup', 'rb'))
    df_unsup=scaler('scaler_unsup')
    df_unsup=one_hot_encode('ohe_unsup',df=df_unsup)
    prediction_unsup=model_unsup.predict(df_unsup)
    grade=prediction_unsup[0]

    #Unpickling supervised clustering model predicting grade
    model_clust=pickle.load(open('model_clust', 'rb'))
    df_clust=scaler('scaler_clust')
    df_clust=one_hot_encode('ohe_clust',df=df_clust)
    prediction_clust=model_clust.predict(df_clust)
    #grade=prediction_clust[0]

    #Getting default loan case
    if prediction_class_def[0]==1:
        loan='Loan will go default'

        #Unpickling classification model predicting zero or non-zero recoveries
        #df=pd.DataFrame.from_records([data])
        model_class_rec = pickle.load(open('model_class_rec', 'rb'))
        df_class_rec=one_hot_encode('ohe_class_rec')
        prediction_class_rec=model_class_rec.predict(df_class_rec)

        #Getting non-zero recoveries case
        if prediction_class_rec[0]==1:

            #Unpickling regression model predicting recoveries
            #df=pd.DataFrame.from_records([data])
            model_reg_rec = pickle.load(open('model_reg_rec', 'rb'))
            df_reg_rec=one_hot_encode('ohe_reg_rec')
            prediction_reg_rec=model_reg_rec.predict(df_reg_rec)
            recoveries=prediction_reg_rec[0]

        #Getting zero recoveries case
        else:
            recoveries='0'
        
        #Getting outcome
        outcome={'Loan Prediction: ':loan,'Recoveries: ':str(recoveries)
        ,'Predicted Interest Rate: ':str(intr_rate), 'Grade':clust[grade],
        'Purpose':str(prediction_nlp)
        } 

    #Getting non-default loan case    
    else:
        loan='Loan will not go default'

        #Getting outcome
        outcome={'Loan Prediction: ':loan,'Predicted Interest Rate: ':str(intr_rate),
        'Grade':clust[grade],'Purpose':str(prediction_nlp)} 

    #Dumping outcome to JSON 
    return json.dumps(outcome)
	
if __name__=='__main__':
    app.run(debug=True)
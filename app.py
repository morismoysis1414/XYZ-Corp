#mports
from flask import Flask, request
import logging
import pickle
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder



logging.basicConfig(filename='dump.log', level=logging.INFO)

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict_loan():
    
    #Getting data from JSON file
    req_j=request.json
    data=req_j
  
    #Creating dataframe
    df=pd.DataFrame.from_records([data])
    df_nlp=df['desc']
    df=df.drop('desc',axis=1)

    model_nlp = pickle.load(open('model_nlp', 'rb'))
    vect_nlp=pickle.load(open('vect_nlp', 'rb'))
    df_nlp=vect_nlp.transform(df_nlp)
    prediction_nlp=model_nlp.predict(df_nlp)[0]
    #prediction_nlp=prediction_nlp[0]

    ohe_cols=['purpose','verification_status','home_ownership',
    'initial_list_status','term'] #address
    df_scale=df.drop(ohe_cols,axis=1)
    df_non_scale=df[ohe_cols]

    clust={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'}


    #Unpickling classification model predicting default index
    model_class_def = pickle.load(open('model_class_def', 'rb'))
    ohe_class_def=pickle.load(open('ohe_class_def', 'rb'))
    df_enc = pd.DataFrame(ohe_class_def.transform(df[ohe_cols]).toarray(),
    index=df.index)
    df_class_def=df.join(df_enc).drop(ohe_cols,axis=1)
    df_class_def.columns = df_class_def.columns.map(str)
    prediction_class_def=model_class_def.predict(df_class_def)

    #Unpickling regression model predicting interest rate
    model_reg_int=pickle.load(open('model_reg_int', 'rb'))
    ohe_reg_int=pickle.load(open('ohe_reg_int', 'rb'))
    df_enc = pd.DataFrame(ohe_reg_int.transform(df[ohe_cols]).toarray(),
    index=df.index)
    df_reg_int=df.join(df_enc).drop(ohe_cols,axis=1)
    df_reg_int.columns = df_reg_int.columns.map(str)
    prediction_reg_int=model_reg_int.predict(df_reg_int)
    intr_rate=prediction_reg_int[0]


    model_unsup=pickle.load(open('model_unsup', 'rb'))
    ohe_unsup=pickle.load(open('ohe_unsup', 'rb'))
    scaler_unsup=pickle.load(open('scaler_unsup', 'rb'))
    df_scaled = pd.DataFrame(scaler_unsup.transform(df_scale),index=df_scale.index,columns=df_scale.columns)
    df_unsup=df_scaled.join(df_non_scale)
    df_enc = pd.DataFrame(ohe_unsup.transform(df[ohe_cols]).toarray(),
    index=df.index)
    df_unsup=df_unsup.join(df_enc).drop(ohe_cols,axis=1)
    df_unsup.columns = df_unsup.columns.map(str)
    prediction_unsup=model_unsup.predict(df_unsup)
    grade_unsup=prediction_unsup[0]

    #Getting default loan case
    if prediction_class_def[0]==1:
        loan='Loan will go default'

        #Unpickling classification model predicting zero or non-zero recoveries
        #df=pd.DataFrame.from_records([data])
        model_class_rec = pickle.load(open('model_class_rec', 'rb'))
        ohe_class_rec=pickle.load(open('ohe_class_rec', 'rb'))
        df_enc = pd.DataFrame(ohe_class_rec.transform(df[ohe_cols]).toarray(),
        index=df.index)
        df_class_rec=df.join(df_enc).drop(ohe_cols,axis=1)
        df_class_rec.columns = df_class_rec.columns.map(str)
        prediction_class_rec=model_class_rec.predict(df_class_rec)

        #Getting non-zero recoveries case
        if prediction_class_rec[0]==1:

            #Unpickling regression model predicting recoveries
            #df=pd.DataFrame.from_records([data])
            model_reg_rec = pickle.load(open('model_reg_rec', 'rb'))
            ohe_reg_rec=pickle.load(open('ohe_reg_rec', 'rb'))
            df_enc = pd.DataFrame(ohe_reg_rec.transform(df[ohe_cols]).toarray(),
            index=df.index)
            df_reg_rec=df.join(df_enc).drop(ohe_cols,axis=1)
            df_reg_rec.columns = df_reg_rec.columns.map(str)
            prediction_reg_rec=model_reg_rec.predict(df_reg_rec)
            recoveries=prediction_reg_rec[0]

        #Getting zero recoveries case
        else:
            recoveries='0'
        
        #Getting outcome
        outcome={'Loan Prediction: ':loan,'Recoveries: ':str(recoveries)
        ,'Predicted Interest Rate: ':str(intr_rate), 'Grade Unsupervised':clust[grade_unsup],
        'Predicted Purpose':str(prediction_nlp)
        } 

    #Getting non-default loan case    
    else:
        loan='Loan will not go default'

        #Getting outcome
        outcome={'Loan Prediction: ':loan,'Predicted Interest Rate: ':str(intr_rate),
        'Grade Unsupervised':clust[grade_unsup],'Predicted Purpose':str(prediction_nlp)} 

    #Dumping outcome to JSON 
    return json.dumps(outcome)
	
if __name__=='__main__':
    app.run(debug=True)
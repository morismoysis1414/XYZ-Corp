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

    #Unpickling classification model predicting default index
    model_class_def = pickle.load(open('model_class_def', 'rb'))
    ohe_cols=['purpose','verification_status','home_ownership','initial_list_status','term'] #address
    ohe_class_def=pickle.load(open('ohe_class_def', 'rb'))
    df_enc = pd.DataFrame(ohe_class_def.transform(df[ohe_cols]).toarray(),index=df.index)
    df_class_def=df.join(df_enc).drop(ohe_cols,axis=1)
    df_class_def.columns = df_class_def.columns.map(str)
    prediction_class_def=model_class_def.predict(df_class_def)

    #Unpickling regression model predicting interest rate
    #df=pd.DataFrame.from_records([data])
    model_reg_int=pickle.load(open('model_reg_int', 'rb'))
    ohe_reg_int=pickle.load(open('ohe_reg_int', 'rb'))
    df_enc = pd.DataFrame(ohe_reg_int.transform(df[ohe_cols]).toarray(),index=df.index)
    df_reg_int=df.join(df_enc).drop(ohe_cols,axis=1)
    df_reg_int.columns = df_reg_int.columns.map(str)
    prediction_reg_int=model_reg_int.predict(df_reg_int)
    intr_rate=prediction_reg_int[0]


    #Getting default loan case
    if prediction_class_def[0]==1:
        loan='Loan will go default'

        #Unpickling classification model predicting zero or non-zero recoveries
        #df=pd.DataFrame.from_records([data])
        model_class_rec = pickle.load(open('model_class_rec', 'rb'))
        ohe_class_rec=pickle.load(open('ohe_class_rec', 'rb'))
        df_enc = pd.DataFrame(ohe_class_rec.transform(df[ohe_cols]).toarray(),index=df.index)
        df_class_rec=df.join(df_enc).drop(ohe_cols,axis=1)
        df_class_rec.columns = df_class_rec.columns.map(str)
        prediction_class_rec=model_class_rec.predict(df_class_rec)

        #Getting non-zero recoveries case
        if prediction_class_rec[0]==1:

            #Unpickling regression model predicting recoveries
            #df=pd.DataFrame.from_records([data])
            model_reg_rec = pickle.load(open('model_reg_rec', 'rb'))
            ohe_reg_rec=pickle.load(open('ohe_reg_rec', 'rb'))
            df_enc = pd.DataFrame(ohe_reg_rec.transform(df[ohe_cols]).toarray(),index=df.index)
            df_reg_rec=df.join(df_enc).drop(ohe_cols,axis=1)
            df_reg_rec.columns = df_reg_rec.columns.map(str)
            prediction_reg_rec=model_reg_rec.predict(df_reg_rec)
            recoveries=prediction_reg_rec[0]

        #Getting zero recoveries case
        else:
            recoveries='0'
        
        #Getting outcome
        outcome={'Loan Prediction: ':loan,'Recoveries: ':str(recoveries),'Predicted Interest Rate: ':str(intr_rate)} 

    #Getting non-default loan case    
    else:
        loan='Loan will not go default'

        #Getting outcome
        outcome={'Loan Prediction: ':loan,'Predicted Interest Rate: ':str(intr_rate)} 

    #Dumping outcome to JSON 
    return json.dumps(outcome)
	
if __name__=='__main__':
    app.run(debug=True)
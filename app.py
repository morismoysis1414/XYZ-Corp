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
    prediction_class_def=model_class_def.predict(df)

    #Unpickling regression model predicting interest rate
    model_reg_int=pickle.load(open('model_reg_int', 'rb'))
    prediction_reg_int=model_reg_int.predict(df)
    intr_rate=prediction_reg_int[0]


    #Getting default loan case
    if prediction_class_def[0]==1:
        loan='Loan will go default'

        #Unpickling classification model predicting zero or non-zero recoveries
        model_class_rec = pickle.load(open('model_class_rec', 'rb'))
        prediction_class_rec=model_class_rec.predict(df)

        #Getting non-zero recoveries case
        if prediction_class_rec[0]==1:

            #Unpickling regression model predicting recoveries
            model_reg_rec = pickle.load(open('model_reg_rec', 'rb'))
            prediction_reg_rec=model_reg_rec.predict(df)
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
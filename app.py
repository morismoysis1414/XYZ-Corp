from flask import Flask, request
import logging
import pickle
import json
import pandas as pd
#import xgboost as xgb

logging.basicConfig(filename='dump.log', level=logging.INFO)

app = Flask(__name__)


@app.route("/predict", methods=["GET"])
def predict_loan():
    
    req_j=request.json
    test_data=req_j
  
    test_df=pd.DataFrame.from_records([test_data])

    #test_df=test_df[['age', 'sex', 'race', 'state_code', 'county_code', 'in_cover_dur',
    #   'out_cover_dur', 'carrier_cover_dur', 'drug_cover_dur', 'alzheimers',
    #   'heart_failure', 'kidney_disease', 'cancer', 'pulmonary_disease',
    #   'depression', 'diabetes', 'ischemic_heart_disease', 'osteoporosis',
    #   'arthritis', 'stroke', 'in_cover_amt', 'in_excess_amt', 'out_cover_amt',
    #   'out_excess_amt']]
    
    model_class_def = pickle.load(open('model_class_def', 'rb'))
    prediction_class_def=model_class_def.predict(test_df)

    model_reg_int=pickle.load(open('model_reg_int', 'rb'))
    prediction_reg_int=model_reg_int.predict(test_df)
    intr_rate=prediction_reg_int[0]



    if prediction_class_def[0]==1:
        loan='Loan will go default'
        model_class_rec = pickle.load(open('model_class_rec', 'rb'))
        prediction_class_rec=model_class_rec.predict(test_df)
        if prediction_class_rec[0]==1:
            model_reg_rec = pickle.load(open('model_reg_rec', 'rb'))
            prediction_reg_rec=model_reg_rec.predict(test_df)
            recoveries=prediction_reg_rec[0]
        else:
            recoveries='0'
        
        outcome={'Loan Prediction: ':loan,'Recoveries: ':str(recoveries),'Predicted Interest Rate: ':str(intr_rate)} 
    else:
        loan='Loan will not go default'
        outcome={'Loan Prediction: ':loan,'Predicted Interest Rate: ':str(intr_rate)} 

    return json.dumps(outcome)
	
if __name__=='__main__':
    app.run(debug=True)
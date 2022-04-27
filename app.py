from flask import Flask, request
import logging
import pickle
import json
import pandas as pd

logging.basicConfig(filename='dump.log', level=logging.INFO)

app = Flask(__name__)


@app.route("/predict", methods=["GET"])
def predict_claim():
    
    req_j=request.json
    test_data=req_j['demographics']
    test_data['age']=int(test_data['age']*365)

    sex={'male':int(1), 'female':int(0),'other':int(0)}
    test_data['sex']=sex[test_data['sex']]
    race={'white':int(1),'black':int(2),'hispanic':int(5),'other':int(3)}
    test_data['race']=race[test_data['race']]
    diseases=['alzheimers','heart_failure', 'kidney_disease', 'cancer', 'pulmonary_disease','depression', 'diabetes', 'ischemic_heart_disease', 'osteoporosis',
       'arthritis', 'stroke']


    covers=['in_cover_dur','out_cover_dur', 'carrier_cover_dur','drug_cover_dur','in_cover_amt', 'in_excess_amt', 'out_cover_amt',
       'out_excess_amt']

    for illness in diseases:
        if illness in req_j['diseases']:
            test_data[illness]=int(1)
        else:
            test_data[illness]=int(0)

    for cover in covers:
        if cover in req_j['cover_info']:
            test_data[cover]=req_j['cover_info'][cover]
        else:
            test_data[cover]=int(0)    
    
    
    test_df=pd.DataFrame.from_records([test_data])

    test_df=test_df[['age', 'sex', 'race', 'state_code', 'county_code', 'in_cover_dur',
       'out_cover_dur', 'carrier_cover_dur', 'drug_cover_dur', 'alzheimers',
       'heart_failure', 'kidney_disease', 'cancer', 'pulmonary_disease',
       'depression', 'diabetes', 'ischemic_heart_disease', 'osteoporosis',
       'arthritis', 'stroke', 'in_cover_amt', 'in_excess_amt', 'out_cover_amt',
       'out_excess_amt']]
    
    model_class = pickle.load(open('model_class', 'rb'))
    prediction_class=model_class.predict(test_df)

    if prediction_class==1:
        claim='Likely to claim'
        model_regr = pickle.load(open('model_regr', 'rb'))
        prediction_regr=model_regr.predict(test_df)
        if prediction_regr==0:
                claim_amt='£0-£50'
        elif prediction_regr==1:
            claim_amt='£50-£250'
        elif prediction_regr==2:
            claim_amt='£250-£1000'
        elif prediction_regr==3:
            claim_amt='£1000-£3500'
        elif prediction_regr==4:
            claim_amt='£3500-£60000'
        outcome = {'claim':claim,'amount':claim_amt}
    else:
        claim='Not likely to claim'
        outcome = {'claim':claim}

    return json.dumps(outcome)
	
if __name__=='__main__':
    app.run(debug=True)
import requests
import cPickle as pickle
from model import MyModel, get_data
from pymongo import MongoClient
import pandas as pd
from utils import preprocess_series
import json

def model_run():
    """
    Load picked model, get real time data from Heroku and make predictions on streaming data.
    Dump new streaming data and predictions into MongoDB
    """
    
    with open('model.pkl') as f:
        model = pickle.load(f)
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    client = MongoClient()
    db = client['fraud']
    collection = db['events']
    stored_id = []
    while True:
        data = requests.get(url).json()
        ob_id = data['object_id']
        if ob_id not in stored_id:
            d = pd.Series(data)
            X = preprocess_series(d)
            prob=model.predict_proba(X)[0][1]
            if prob < .5:
                risk = 'No'
            elif prob < .6:
                risk = 'Low'
            elif prob < .8:
                risk = 'Medium'
            else:
                risk = 'High'
            stored_id.append(ob_id)
            data['fraud'] = int(prob>=.5)
            data['prob'] = round(prob,2)
            data['risk'] = risk
            collection.insert_one(data)
if __name__ == '__main__':
    model_run()

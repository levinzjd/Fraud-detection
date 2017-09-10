from flask import Flask
from flask import render_template, request
import requests
import cPickle as pickle
from model import MyModel, get_data
import pandas as pd
from utils import preprocess_series
import json
import time
from pymongo import MongoClient

"""
Flask web app for streaming fraud detection
"""

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hello():
    return render_template('index.html')

@app.route('/model_summery',methods=['POST'])
def model_summery():
    client = MongoClient()
    db = client['fraud']

    collection = db['events']
    all_prob_fraud=[]
    all_risk_fraud=[]
    # for ducument in collection.find():
    total_instances = collection.count()
    total_fraud = collection.find({'fraud': 1}).count()
    percentage_fraud = total_fraud*100.0/total_instances
    total_hr = collection.find({'risk': 'High'})
    total_mr = collection.find({'risk': 'Medium'})
    total_lr = collection.find({'risk': 'Low'})


    return render_template('result.html',total = total_instances,fraud=total_fraud, per=percentage_fraud)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)

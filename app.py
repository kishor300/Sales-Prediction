# flask --app app.py --debug run

from Model import Prediction
from flask import Flask, request, jsonify, render_template 

import pickle
import pandas as pd
from datetime import datetime, timedelta
from dateutil import relativedelta

# Create a Flask app
app = Flask(__name__, static_url_path='/static')

# Load the pickled machine learning model
with open('SARIMA_SALES_PREDICTION.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for the home page

@app.route('/')
def home():
    response = "Homepage"
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sales():

    # Read start and end dates from POST request
    
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    result = Prediction(start_date, end_date)
    
    return render_template('future_prediction.html', context = result)

if __name__ == '__main__':
    app.run(debug=True)

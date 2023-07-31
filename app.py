# flask --app app.py --debug run

from Model import Prediction
from flask import Flask, request, jsonify, render_template 
# request-contain all data sent from client to server
# jsonify- convert Python objects into JSON-formatted responses
# JSON (JavaScript Object Notation) lightweight data interchange format
# render_template - render HTML templates and return them as responses to the client

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


# @app.route('/index2.html')
# def index2():
#     return render_template('index2.html')

# Route to serve the image file


@app.route('/predict', methods=['POST'])
def predict_sales():

        # Read start and end dates from POST request
    # start_date = request.form['start_date']
    # end_date = request.form['end_date']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    # print("date ****************************************", start_date, end_date)
    # data = request.get_json()  # get the JSON data from the request body
    # do something with the data
    # print(data)
    # response_data = {"message": "Data received successfully"}
    # return jsonify(response_data)
    # start_date = data['start_date']
    # end_date = data['end_date']
    result = Prediction(start_date, end_date)
    # print('#############################################333')
    # print(result)
    # Convert dates to pandas datetime objects and format as 'year-month-date'
    # start_date = datetime.strptime(start_date, '%Y-%m-%d')
    # end_date = datetime.strptime(end_date, '%Y-%m-%d')
    ############################################    MILIND ####################################
    # pd_start_date = pd.Timestamp(start_date[:-2]+('01 00:00:00'))   # date for creting dataframe
    # future_dates = [ df_stationarity.index[-1]+ DateOffset(months=x)for x in range(0,months) ]
    # Predict sales using the model
    # y_pred = model.predict(start= 48, end=48+months, dynamic=True)
    # print(y_pred.tolist())
    # Return predicted sales as JSON response
    # start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    # end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    # dates_between = get_dates_between(start_date, end_date)
    # print(dates_between)  # Output: ['2023-05-01', '2023-05-02', '2023-05-03', '2023-05-04', '2023-05-05', '2023-05-06', '2023-05-07', '2023-05-08', '2023-05-09', '2023-05-10']
    # response = {'predicted_sales': result}
    # content = {'thing': 'some stuff',
    #            'other': 'more stuff'}
    # # dic = {str(i)[:10]: j for i, j in zip(result.index(), y_pred.values)}
    # predict_sales = {"2023-06-01" : 460523.4820854854}
    return render_template('future_prediction.html', context = result)




if __name__ == '__main__':
    app.run(debug=True)

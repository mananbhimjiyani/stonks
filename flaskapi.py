from flask import Flask, request, jsonify
import pandas as pd
import joblib
import real_time_sentiment as rts
import yfinance as yf
import nltk
import numpy as np
import warnings
import sys
print(sys.executable)
warnings.filterwarnings("ignore", category=DeprecationWarning)

nltk.download('vader_lexicon')

app = Flask(__name__)
from flask_cors import CORS

CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/predict_forex": {"origins": "http://localhost:3000"}})


# Define the company mapping
company_mapping = {
    0: 'ADANIPORTS.NS', 1: 'APOLLOHOSP.NS', 2: 'ASIANPAINT.NS', 3: 'AXISBANK.NS',
    4: 'BAJAJ-AUTO.NS', 5: 'BAJAJFINSV.NS', 6: 'BPCL.NS', 7: 'BRITANNIA.NS', 8: 'CIPLA.NS', 9: 'COALINDIA.NS',
    10: 'DIVISLAB.NS', 11: 'DRREDDY.NS', 12: 'EICHERMOT.NS', 13: 'GRASIM.NS', 14: 'HCLTECH.NS', 15: 'HDFCLIFE.NS',
    16: 'HDFCBANK.NS', 17: 'HEROMOTOCO.NS', 18: 'HINDALCO.NS', 19: 'HINDUNILVR.NS', 20: 'ICICIBANK.NS',
    21: 'INDUSINDBK.NS', 22: 'INFY.NS', 23: 'ITC.NS', 24: 'JIOFIN.NS', 25: 'JSWSTEEL.NS', 26: 'KOTAKBANK.NS',
    27: 'LT.NS', 28: 'LTIM.NS', 29: 'M&M.NS', 30: 'MARUTI.NS', 31: 'NESTLEIND.NS', 32: 'NIFTY50.NS', 33: 'NTPC.NS',
    34: 'ONGC.NS', 35: 'POWERGRD.NS', 36: 'RELIANCE.NS', 37: 'SBILIFE.NS', 38: 'SBIN.NS', 39: 'SUNPHARMA.NS',
    40: 'TCS.NS', 41: 'TATACONSUM.NS', 42: 'TATAMOTORS.NS', 43: 'TATASTEEL.NS', 44: 'TECHM.NS', 45: 'TITAN.NS',
    46: 'ULTRACEMCO.NS', 47: 'UPL.NS', 48: 'WIPRO.NS'
}

# Define the Forex mapping
forex_mapping = {
    0: 'AUD-USD-ASK.joblib', 1: 'AUD-USD-BID.joblib', 2: 'EUR-USD-ASK.joblib', 3: 'EUR-USD-BID.joblib',
    4: 'GBP-USD-ASK.joblib', 5: 'GBP-USD-BID.joblib', 6: 'NZD-USD-ASK.joblib', 7: 'NZD-USD-BID.joblib',
    8: 'USD-CAD-ASK.joblib', 9: 'USD-CAD-BID.joblib', 10: 'USD-CHF-ASK.joblib', 11: 'USD-CHF-BID.joblib',
    12: 'USD-JPY-ASK.joblib', 13: 'USD-JPY-BID.joblib', 14: 'XAG-USD-ASK.joblib', 15: 'XAG-USD-BID.joblib'
}

# Load the stock model

@app.route('/predict', methods=['GET'])
def predict():
    model = joblib.load('stock_price_predictor_model.joblib')  # Replace with your model filename
    company_symbol = request.args.get('company_symbol')

    # Validate and convert the company_symbol to an integer
    try:
        company_symbol = int(company_symbol)
        if company_symbol not in company_mapping:
            return jsonify({"error": "Invalid company_symbol."})
    except ValueError:
        return jsonify({"error": "Invalid company_symbol. It should be an integer."})

    company = company_mapping[company_symbol]

    # Prepare the input data for stock prediction
    data = pd.DataFrame({
        'Close_Lagged': [rts.get_current_closing(company)],
        'Sentiment_Score': [rts.get_current_sentiment(company)],
        'Company': [company_symbol],
    })

    # Create the test DataFrame
    test_data = pd.DataFrame(data)

    # Extract features from test data
    X_test = test_data[['Close_Lagged', 'Sentiment_Score', 'Company']]

    # Make predictions using the loaded stock model
    y_pred = model.predict(X_test)

    # Return the stock prediction as JSON
    response = jsonify({"prediction": float(y_pred[0])})
    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

@app.route('/predict_forex', methods=['GET'])
def predict_forex():
    forex_symbol = int(request.args.get('forex_symbol'))
    # Validate and handle the forex index
    if forex_symbol not in forex_mapping:
        return jsonify({"error": "Invalid forex_index."})
    model = joblib.load(forex_mapping[forex_symbol])
    forex = yf.Ticker(forex_mapping[forex_symbol][:3] + '=X')
    data = forex.history(period="1d", interval="1m")

    # Extract necessary data for prediction
    forex_close = data['Close'].iloc[-1]
    forex_high = data['High'].iloc[-1]
    forex_low = data['Low'].iloc[-1]
    forex_volume = data['Volume'].iloc[-1]
    print(forex_close)

    # Make the Forex prediction
    forex_prediction = model.predict(pd.DataFrame([{
        'Close': forex_close,
        'High': forex_high,
        'Low': forex_low,
        'Volume': forex_volume
    }]))

    formatted_forex_prediction = "{:.2f}".format(float(forex_prediction[0]))

    # Return the Forex prediction as JSON
    response = jsonify({"prediction": formatted_forex_prediction})
    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

if __name__ == '__main__':
    from gunicorn.app.wsgiapp import WSGIApplication
    app_wsgi = WSGIApplication()
    app_wsgi.app_uri = 'flaskapi:app'
    app_wsgi.run()
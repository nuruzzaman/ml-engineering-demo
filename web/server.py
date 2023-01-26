import logging
import joblib
import ast
import numpy as np
from logging.handlers import TimedRotatingFileHandler

from flask import Flask, request, render_template, jsonify


def init_log(log_file='log/info.log'):
    handler = TimedRotatingFileHandler(log_file, when="D", interval=1, backupCount=7)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


logger = init_log()
# set the project root directory as the static folder
app = Flask(__name__, static_url_path='')


@app.route('/', methods=['GET', 'POST'])
def default_api():
    return render_template('index.html')


@app.route('/stream', methods=['POST'])
def stream():
    # Get the payload from the request
    payload = request.get_json()

    # Extract the input feature from the payload
    X_input = payload['data']
    X = int(X_input)

    # Load model
    load_model = joblib.load('model/Diabetes.pkl', 'rb')

    # Make prediction on the given data
    y_pred = load_model.predict([X])
    mgs = f"The prediction of {X} is {y_pred[0]:.3f}"
    print(f'Y value is: {y_pred}, when X is {X}')
    return mgs


@app.route('/batch', methods=['POST'])
def batch():
    # Get the payload from the request
    payload = request.get_json()

    # Load model
    load_model = joblib.load('model/Diabetes.pkl', 'rb')

    # Extract the input feature from the payload
    X_input = payload['data']
    X = ast.literal_eval(X_input)

    pred_list = []
    if len(X)>0:
        for i in X:
            # Make prediction on the given data
            y_pred = load_model.predict([i])
            print(f'Y value is: {y_pred}, when X is {i}')
            pred_list.append(y_pred[0])

    # convert list to string
    string_list = ["{:.3f}".format(i) for i in pred_list]
    final_string = '\n'.join(string_list)
    return final_string


class SimpleLinearRegression:
    def predict(self, X):
        y_hat = np.dot(X, self.W.T) + self.b
        return y_hat


if __name__ == '__main__':
    print("Server started at port 5000")
    app.run('127.0.0.1', port=5000, debug=True)

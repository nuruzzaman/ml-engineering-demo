import logging
import pickle
import ast
import numpy as np
from logging.handlers import TimedRotatingFileHandler

from flask import Flask, request, render_template, jsonify


# set the project root directory as the static folder
app = Flask(__name__, static_url_path='')

# Load the model from a file
model_filename = "model/Diabetes.pkl"
with open(model_filename, "rb") as file:
    load_model = pickle.load(file)


def init_log(log_file='log/info.log'):
    handler = TimedRotatingFileHandler(log_file, when="D", interval=1, backupCount=7)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

logger = init_log()


@app.route('/')
def default_api():
    return render_template('index.html')


@app.route('/stream', methods=['POST'])
def stream():
    # Get the payload from the request
    payload = request.get_json()

    # Extract the input feature from the payload
    X_input = payload['data']
    X = int(X_input)

    # Make prediction on the given data
    y_pred = load_model.predict([[X]])
    y = y_pred[0]
    final_predict_mgs = f"The prediction of {X} is: {y[0]:.3f}"
    return final_predict_mgs


@app.route('/batch', methods=['POST'])
def batch():
    # Get the payload from the request
    payload = request.get_json()

    # Extract the input feature from the payload and convert into `np.array`
    X_input = payload['data']
    X = ast.literal_eval(X_input)
    X = np.array([[i] for i in X])

    # Make prediction on the input data
    pred_list = load_model.predict(X)

    string_list = []
    for x, i in zip(X, pred_list):
        string_list.append("Prediction of {} is: {:.3f}\n".format(x[0], i[0]))
    final_predict_mgs = ''.join(string_list)
    return final_predict_mgs


if __name__ == '__main__':
    print("ML engineering server started at port 5000")
    app.debug = False

    from werkzeug.serving import run_simple
    run_simple("localhost", 5000, app)

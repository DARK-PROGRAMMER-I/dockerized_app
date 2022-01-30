import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pickle
import flask

import io
import string

import time
import os

# import tensorflow as tf
from flask import Flask,render_template, jsonify, request
import requests
# ### Unpickle the Model

churn_model= pickle.load(open("data/churn_model.pkl", "rb"))
# ### Reading test data for Confirmation Purpose
X_test= pd.read_csv("data/X_test.csv")

# ### Loading Predictions

predictions= np.loadtxt("data/predictions.csv")

# ### Testing Model

churn_model.predict(X_test)


# ### Preparing Flask Model

app= Flask(__name__)
@app.route("/")
def hellow_world():
    return render_template('templates/churn_templete.html')

# All kind of errors checking will go here
@app.route("/predict", methods= ["POST", "GET"])
def make_predict():
    print(request.form)

    # Convert out json to a numpy array.
    # predict_request= [data["is_male"], data["num_inters"], data["late_on_payment"], data["age"], data["years_in_contract"]]
    int_features= [int(x) for x in request.form.values()]
    # predict_request= np.array(predict_request)
    final= np.array(int_features)
    print(int_features)
    print(final)
    # np array will go into Model and results will be predicted
    # y_hat= churn_model.predict(predict_request)
    prediction=  churn_model.predict_proba(final)
    # output= [y_hat[0]]
    output= "0: .{1}f".format(prediction[0][1], 2)

    if output >str( 0.5):
        return render_template('churn_templete.html', pred= 'The probablity of getting hired is{ }'.format(output))
    else:
        render_template('churn_templete.html', pred= 'The Probablity of not getting hired is { }'.format(output) )


    return jsonify(results= output)

if __name__ == "__main__":
    app.run(host='0.0.0.0')






















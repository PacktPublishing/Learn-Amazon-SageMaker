#!/usr/bin/env python

import joblib, os
import pandas as pd
from io import StringIO

import flask
from flask import Flask, Response

model_dir = '/opt/ml/model'
model = joblib.load(os.path.join(model_dir, "model.joblib"))

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="\n", status=200)

@app.route("/invocations", methods=["POST"])
def predict():
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        print("input: ", s.getvalue())
        data = pd.read_csv(s, header=None)
        response = model.predict(data)
        response = str(response)
        print("response: ", response)
    else:
        return flask.Response(response='CSV data only', status=415, mimetype='text/plain')

    return Response(response=response, status=200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

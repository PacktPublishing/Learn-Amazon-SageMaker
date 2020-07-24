#!/usr/bin/env python

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os, json

if __name__ == '__main__':
    
    config_dir = '/opt/ml/input/config'
    training_dir = '/opt/ml/input/data/training'
    model_dir = '/opt/ml/model'
    
    with open(os.path.join(config_dir, 'hyperparameters.json')) as f:
        hp = json.load(f)
        print(hp)
        normalize = hp['normalize']
        test_size = float(hp['test-size'])
        random_state = int(hp['random-state'])
        
    filename = os.path.join(training_dir, 'housing.csv')
    data = pd.read_csv(filename)
    labels = data[['medv']]
    samples = data.drop(['medv'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, 
                                                        test_size=test_size, random_state=random_state)
    regr = LinearRegression(normalize=normalize)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    
    joblib.dump(regr, os.path.join(model_dir, 'model.joblib'))
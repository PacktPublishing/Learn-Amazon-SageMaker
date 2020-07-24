
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import argparse, os

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--random-state', type=int, default=123)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args, _ = parser.parse_known_args()
    normalize = args.normalize
    test_size = args.test_size
    random_state = args.random_state
    model_dir  = args.model_dir
    training_dir = args.training

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
    
    model = os.path.join(model_dir, 'model.joblib')
    joblib.dump(regr, model)
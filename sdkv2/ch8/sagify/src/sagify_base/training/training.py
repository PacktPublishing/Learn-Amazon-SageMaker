import json, os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train(input_data_path, model_save_path, hyperparams_path=None):
    """
    The function to execute the training.

    :param input_data_path: [str], input directory path where all the training file(s) reside in
    :param model_save_path: [str], directory path to save your model(s)
    :param hyperparams_path: [optional[str], default=None], input path to hyperparams json file.
    Example:
        {
            "max_leaf_nodes": 10,
            "n_estimators": 200
        }
    """
    with open(hyperparams_path) as f:
        hp = json.load(f)
        test_size =  float(hp['test-size'])
        random_state =  int(hp['random-state'])

    data = pd.read_csv(os.path.join(input_data_path, 'bank-additional-full.csv'))
    data = pd.get_dummies(data)
    data.drop(['y_no'], axis=1, inplace=True)

    x = data.drop(['y_yes'], axis=1)
    y = data['y_yes']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_test.to_csv(os.path.join(model_save_path, 'x_test.csv'), index=False,header=False)
    y_test.to_csv(os.path.join(model_save_path, 'y_test.csv'), index=False,header=False)

    cls = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc')
    cls.fit(x_train, y_train)
    auc = cls.score(x_test, y_test)

    accuracy_report_file_path = os.path.join(model_save_path, 'report.txt')
    with open(accuracy_report_file_path, 'w') as f:
        f.write(str(auc))

    cls.save_model(os.path.join(model_save_path, 'model.joblib'))





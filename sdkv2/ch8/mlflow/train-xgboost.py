import mlflow.xgboost
import xgboost as xgb
from load_dataset import load_dataset

if __name__ == '__main__':
    mlflow.set_experiment('dm-xgboost')
    with mlflow.start_run(run_name='dm-xgboost-basic') as run:
        x_train, x_test, y_train, y_test = load_dataset(
            'bank-additional/bank-additional-full.csv', ';'
        )
        cls = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc')
        cls.fit(x_train, y_train)
        auc = cls.score(x_test, y_test)
        print('AUC ', auc)
        mlflow.log_metric('auc', auc)

        mlflow.xgboost.log_model(cls, 'dm-xgboost-model')
        mlflow.end_run()


from sklearn.metrics import mean_squared_error, r2_score

def printmetrics(y_test, y_pred):
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
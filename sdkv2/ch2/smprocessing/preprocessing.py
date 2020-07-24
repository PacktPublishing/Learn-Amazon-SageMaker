import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    split_ratio = args.train_test_split_ratio

    # Load dataset into a pandas dataframe
    input_data_path = os.path.join('/opt/ml/processing/input', 'bank-additional-full.csv')
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)
    
    # Remove lines with missing values and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Count samples in the two classes
    one_class = df[df['y']=='yes']
    one_class_count = one_class.shape[0]
    print("Positive samples: %d" % one_class_count)
    zero_class = df[df['y']=='no']
    zero_class_count = zero_class.shape[0]
    print("Negative samples: %d" % zero_class_count)
    zero_to_one_ratio = zero_class_count/one_class_count
    print("Ratio: %.2f" % zero_to_one_ratio)
        
    # Add a new column to flag customers who have never been contacted
    df['no_previous_contact'] = np.where(df['pdays'] == 999, 1, 0)
    
    # Add a new column to flag customers who don't have a full time job
    df['not_working'] = np.where(np.in1d(df['job'], ['student', 'retired', 'unemployed']), 1, 0)
        
    print('Splitting data into train and test sets with ratio {}'.format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('y', axis=1), 
        df['y'], 
        test_size=split_ratio, random_state=0)

    preprocess = make_column_transformer(
        (['age', 'duration', 'campaign', 'pdays', 'previous'], StandardScaler()),
        (['job', 'marital', 'education', 'default', 'housing', 'loan',
          'contact', 'month', 'day_of_week', 'poutcome'], OneHotEncoder(sparse=False))
    )
    print('Running preprocessing and feature engineering transformations')
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)
    
    print('Train data shape after preprocessing: {}'.format(train_features.shape))
    print('Test data shape after preprocessing: {}'.format(test_features.shape))
    
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
    train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_labels.csv')
    
    test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features.csv')
    test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_labels.csv')
    
    print('Saving training features to {}'.format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
    
    print('Saving test features to {}'.format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)
    
    print('Saving training labels to {}'.format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)
    
    print('Saving test labels to {}'.format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)
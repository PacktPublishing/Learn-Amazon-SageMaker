import argparse, os, subprocess, sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
if __name__=='__main__':
    
    install('nltk')
    import nltk

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--num-reviews', type=int, default=None)
    parser.add_argument('--split-ratio', type=float, default=0.1)
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    filename = args.filename
    num_reviews = args.num_reviews
    split_ratio = args.split_ratio

    # Load dataset into a pandas dataframe
    input_data_path = os.path.join('/opt/ml/processing/input', filename)
    print('Reading input data from {}'.format(input_data_path))
    data = pd.read_csv(input_data_path, sep='\t', compression='gzip',
                       error_bad_lines=False, dtype='str')
    
    # Remove lines with missing values and duplicates
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    
    if num_reviews is not None:
        data = data[:num_reviews]
        
    # Drop unwanted columns
    data = data.drop(['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title',
                  'product_category', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 
                  'review_headline', 'review_date'], axis=1)
    
    # Add label column
    data['label'] = data.star_rating.map({
        '1': '__label__bad__',
        '2': '__label__bad__',
        '3': '__label__medium__',
        '4': '__label__good__',
        '5': '__label__good__'}
    )
    data = data.drop(['star_rating'], axis=1)

    # Move label column to the front
    data = pd.concat([data['label'], data.drop(['label'], axis=1)], axis=1)
    
    # Tokenize reviews
    nltk.download('punkt')
    print('Tokenizing reviews')
    data['review_body'] = data['review_body'].apply(nltk.word_tokenize)
    data['review_body'] = data.apply(lambda row: " ".join(row['review_body']).lower(), axis=1)
    
    # Process data
    print('Splitting data with ratio {}'.format(split_ratio))
    training, validation = train_test_split(data, test_size=split_ratio)
    
    training_output_path = os.path.join('/opt/ml/processing/train', 'training.txt')    
    validation_output_path = os.path.join('/opt/ml/processing/validation', 'validation.txt')
    
    print('Saving training data to {}'.format(training_output_path))
    np.savetxt(training_output_path, training.values, fmt='%s')
    
    print('Saving validation data to {}'.format(validation_output_path))
    np.savetxt(validation_output_path, validation.values, fmt='%s')
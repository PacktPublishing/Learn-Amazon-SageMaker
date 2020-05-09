import io, argparse, os, string, subprocess, sys, pickle

import boto3
import pandas as pd
import numpy as np

from scipy.sparse import lil_matrix

def process_text(text):
    for p in string.punctuation:
        text = text.replace(p, '')
    text = text.lower().split()
    return text

def add_row_to_matrix(line, row):
    for token_id, token_count in row['tokens']:
        token_matrix[line, token_id] = token_count
    return

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
if __name__=='__main__':
    
    install('gensim')
    import gensim
    from gensim import corpora
    
    install('sagemaker')
    import sagemaker.amazon.common as smac
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenames', type=str)
    parser.add_argument('--num-reviews', type=int, default=100000)
    parser.add_argument('--output', type=str)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    filenames = args.filenames
    num_reviews = args.num_reviews
    output = args.output

    data = pd.DataFrame()
        
    # Load dataset into a pandas dataframe
    filenames = filenames.split(',')
    for f in filenames:
        f = 'amazon_reviews_us_{}_v1_00.tsv.gz'.format(f)
        input_data_path = os.path.join('/opt/ml/processing/input', f)
        print('Reading input data from {}'.format(input_data_path))
        tmp = pd.read_csv(input_data_path, sep='\t', compression='gzip',
                          error_bad_lines=False, dtype='str', nrows=num_reviews)
        data = pd.concat([data, tmp])
    
    # Remove lines with missing values and duplicates
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    
    # Drop unwanted columns
    data = data.drop(['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title',
                  'product_category', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 
                  'review_headline', 'review_date', 'star_rating'], axis=1)
    
    # Clean and split reviews into words
    data['review_body'] = data['review_body'].apply(process_text)

    # Build the vocabulary
    dictionary = corpora.Dictionary(data['review_body'])
    print(dictionary)
    
    # Convert each review to a bag of words
    data['tokens'] = data.apply(lambda row: dictionary.doc2bow(row['review_body']), axis=1)
    data = data.drop(['review_body'], axis=1)
    
    # Initialize the sparse matrix
    num_lines = data.shape[0]
    num_columns = len(dictionary)
    token_matrix = lil_matrix((num_lines, num_columns)).astype('float32')
    
    print('Filling word matrix, %d lines %d columns ' % (num_lines, num_columns))
    
    # Fill the matrix with word frequencies
    line = 0
    for _, row in data.iterrows():
        add_row_to_matrix(line, row)
        line+=1   # Can't use indexes, as they may be larger than num_lines
    
    # Write the matrix to protobuf
    buf = io.BytesIO()
    smac.write_spmatrix_to_sparse_tensor(buf, token_matrix, None)
    buf.seek(0)
    
    training_output_path = os.path.join('/opt/ml/processing/train/', 'training.protobuf')
    print('Saving training data to {}'.format(training_output_path))
    with open(training_output_path, 'wb') as f:
        f.write(buf.getbuffer())
    
    dictionary_output_path = os.path.join('/opt/ml/processing/train/', 'dictionary.pkl')
    print('Saving dictionary to {}'.format(dictionary_output_path))
    with open(dictionary_output_path, 'wb') as f:
        pickle.dump(dictionary, f)
    
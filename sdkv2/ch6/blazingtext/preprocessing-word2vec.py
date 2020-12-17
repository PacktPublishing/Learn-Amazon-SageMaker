import argparse, os, subprocess, sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__=='__main__':
    
    install('spacy==2.2.4')
    install('https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz')
    import spacy

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--num-reviews', type=int)
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    filename = args.filename
    num_reviews = args.num_reviews

    # Load dataset into a pandas dataframe
    input_data_path = os.path.join('/opt/ml/processing/input', filename)
    print('Reading input data from {}'.format(input_data_path))
    data = pd.read_csv(input_data_path, sep='\t', compression='gzip',
                       error_bad_lines=False, dtype='str', nrows=num_reviews)
    
    # Remove lines with missing values and duplicates
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
            
    # Drop unwanted columns
    data = data[['review_body']]
     
    # Tokenize reviews
    spacy_nlp = spacy.load('en_core_web_sm')
    def tokenize(text):
        tokens = spacy_nlp.tokenizer(text)
        tokens = [ t.text for t in tokens ]
        return " ".join(tokens).lower()

    print('Tokenizing reviews')
    data['review_body'] = data['review_body'].apply(tokenize)
       
    training_output_path = os.path.join('/opt/ml/processing/train', 'training.txt')    
    
    print('Saving training data to {}'.format(training_output_path))
    np.savetxt(training_output_path, data.values, fmt='%s')
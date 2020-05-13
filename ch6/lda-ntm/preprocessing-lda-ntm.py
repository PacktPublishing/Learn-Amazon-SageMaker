import io, argparse, os, string, subprocess, sys, pickle, string

import boto3
import pandas as pd
import numpy as np

from scipy.sparse import lil_matrix

def process_text(text):
    for p in string.punctuation:
        text = text.replace(p, '')
    text = ''.join([c for c in text if not c.isdigit()])
    text = text.lower().split()
    text = [w for w in text if not w in stop_words] 
    text = [wnl.lemmatize(w) for w in text]
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
    
    install('nltk')
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer 
    stop_words = stopwords.words('english')
    wnl = WordNetLemmatizer()
    
    install('sagemaker')
    import sagemaker.amazon.common as smac
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--num-headlines', type=int, default=1000000)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    filename = args.filename
    num_headlines = args.num_headlines

    data = pd.DataFrame()
        
    # Load dataset into a pandas dataframe
    input_data_path = os.path.join('/opt/ml/processing/input', filename)
    print('Reading input data from {}'.format(input_data_path))
    data = pd.read_csv(input_data_path, compression='gzip',
                      error_bad_lines=False, dtype='str', nrows=num_headlines)
    data.head()
    
    #Shuffle and drop date column
    data = data.sample(frac=1)
    data = data.drop(['publish_date'], axis=1)
    
    # Remove lines with missing values and duplicates
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
        
    # Clean and tokenize
    data['headline_text'] = data['headline_text'].apply(process_text)
    
    # Build the vocabulary
    dictionary = corpora.Dictionary(data['headline_text'])
    dictionary.filter_extremes(keep_n=512)
    print(dictionary)
      
    # Convert each headline to a bag of words
    data['tokens'] = data.apply(lambda row: dictionary.doc2bow(row['headline_text']), axis=1)
    data = data.drop(['headline_text'], axis=1)
    
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
    
    vocabulary_output_path = os.path.join('/opt/ml/processing/train/', 'vocab.txt')
    with open(vocabulary_output_path, 'w') as f:
        for index in range(0,len(dictionary)):
            f.write(dictionary.get(index)+'\n')
    
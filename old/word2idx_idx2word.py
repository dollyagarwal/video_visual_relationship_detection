import json
import pandas as pd
from config import *
import os

def generate():
    vocab = pd.read_csv('{}/vocab.csv'.format(location))
    word2idx={}
    idx2word={}
    for index,row in vocab.iterrows():
      word2idx[row['word']]=row['index']
      idx2word[row['index']]=row['word']

    with open(os.path.join(location,'word2idx.json'), 'w') as fp:
         json.dump(word2idx, fp)

    with open(os.path.join(location,'idx2word.json'), 'w') as fp:
         json.dump(idx2word, fp)

if __name__ == "__main__":

    generate()
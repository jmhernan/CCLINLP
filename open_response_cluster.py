from gensim.models import Word2Vec
  
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
  
from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib.pyplot as plt
#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import re
import sys

from pathlib import Path
import matplotlib.pyplot as plt
import random

import gensim
from gensim.models import Word2Vec

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]

sys.path.append(project_root)

import preprocess as tp

data_path = project_root + '/data/'
updated_responses = sorted(list(filter(lambda x: '.csv' in x, os.listdir(data_path))))[-1]

# data
survey_df = pd.read_csv(os.path.join(data_path, updated_responses), delimiter = ',')

survey_df.columns
survey_df.shape

# Get sentences
text = survey_df['Open-Ended Response_2'].astype(str).replace({'nan': np.NaN}).dropna()
text = text.apply(tp.clean_text)
text.to_list()

def tokenize_text(text):
    """ A function to lower and tokenize text data """ 
    # Lower the text
    lower_text = text.lower()
    # tokenize the text into a list of words
    tokens = nltk.tokenize.word_tokenize(lower_text)
    return tokens

def prepare_text_for_lda(text):
    tokens = tokenize_text(text)
    tokens = [token for token in tokens if len(token) >= 2]
    return tokens

text_data = []

for line in text:
    tokens = prepare_text_for_lda(line)
    if random.random() > .99:
        print(tokens)
    text_data.append(tokens)

# clean up empty lists 
clean_text = [x for x in text_data if x != []]

# TO DO: Clean up words like "nothing"

model = Word2Vec(clean_text, min_count=1)

def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw

l = []
for i in clean_text:
    l.append(sent_vectorizer(i, model))

X = np.array(l)

k_matrix = []

for i in range(1,4):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    k_matrix.append(kmeans.inertia_)

plt.plot(range(1,4), k_matrix)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

n_cluster = 4 

clf = KMeans(n_clusters=n_cluster,
    max_iter=100,
    init='k-means++',
    n_init=1)

labels = clf.fit_predict(X)
print(labels)
for index, clean_text in enumerate(clean_text):
    print(str(labels[index]) + ':' + str(clean_text))



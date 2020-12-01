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
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

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
raw_text = text.apply(tp.clean_text)
raw_text.to_list()

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

for line in raw_text:
    tokens = prepare_text_for_lda(line)
    if random.random() > .99:
        print(tokens)
    text_data.append(tokens)

# clean up empty lists 
clean_text = [x for x in text_data if x != []]

# TO DO: Clean up words like "nothing"

model = Word2Vec(clean_text, min_count=1)
model2 = gensim.models.KeyedVectors.load_word2vec_format('/Users/josehernandez/Documents/eScience/GoogleNews-vectors-negative300.bin', binary=True)

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

def response_vector(word2vec_model, doc):
    doc = [word for word in doc if word in model2.vocab]
    return np.mean(model2[doc], axis=0)

test = response_vector(model2, ['more',
  'information',
  'about',
  'apprenticeships',
  'and',
  'trade',
  'programs',
  'it',
  'would',
  'be',
  'nice',
  'to',
  'add',
  'this',
  'information',
  'with',
  'conversations',
  'about',
  'post',
  'secondary',
  'options'])


def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] 
    return doc


def vector_in_model(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

corpus = [preprocess(text) for text in raw_text]
len(corpus)
clean_text = [x for x in corpus if x != []]
len(clean_text)

w2v_res = [entry for entry in clean_text if vector_in_model(model2, entry)]

x = []
for doc in w2v_res: # append the vector for each document
    x.append(response_vector(model2, doc))
    
X = np.array(x) # list to array

k_matrix = []

for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    k_matrix.append(kmeans.inertia_)

plt.plot(k_matrix)

n_cluster = 5

clf = KMeans(n_clusters=n_cluster,
    max_iter=100,
    init='k-means++',
    n_init=1)

labels = clf.fit_predict(X)

print(labels)

for index, i in enumerate(w2v_res):
    print(str(labels[index]) + ':' + str(i))

len(labels)
len(clean_text)

test = pd.DataFrame(w2v_res,labels)


s1 = 'information about apprenticeships trade programs would nice add information conversations post secondary options'
s2 = 'information about alternative options and college as well as apprenticeships'

#calculate distance between two sentences using WMD algorithm
distance = model2.wmdistance(s1, s2)

print ('distance = %.3f' % distance)

Z = hierarchy.linkage(X, 'ward')
dn = hierarchy.dendrogram(Z)

h_cluster = AgglomerativeClustering(n_clusters = 20, affinity = 'euclidean', linkage = 'ward')

y_h_cluster = h_cluster.fit_predict(X)

print(y_h_cluster)

test = pd.DataFrame(w2v_res,y_h_cluster)
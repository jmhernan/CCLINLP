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

text = survey_df['Open-Ended Response_2'].astype(str).replace({'nan': np.NaN}).dropna()
text = text.apply(tp.clean_text)
text.to_list()
tp.get_top_n_words(text, n=500)

# funs
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def tokenize_text(text):
    """ A function to lower and tokenize text data """ 
    # Lower the text
    lower_text = text.lower()
    # tokenize the text into a list of words
    tokens = nltk.tokenize.word_tokenize(lower_text)
    return tokens

en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize_text(text)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

text_data = []

for line in text:
    tokens = prepare_text_for_lda(line)
    if random.random() > .99:
        print(tokens)
        text_data.append(tokens)

from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')

topics = ldamodel.print_topics(num_words=10)

for topic in topics:
    print(topic)

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

import pyLDAvis.gensim

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_display)

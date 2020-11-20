# Clean up functions
import re
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize

# regex conditions for text cleanup
BAD_SYMBOLS_RE = re.compile(r'[\W]')
REM_LETTER = re.compile(r'(\b\w{1}\b)')

# handle json
def get_metadata_dict(metadata_file):
    metadata_handle = open(metadata_file)
    metadata = json.loads(metadata_handle.read())
    return metadata

# clean text 
def clean_text(text):    
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = REM_LETTER.sub('', text) 
    return text

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
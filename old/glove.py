# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:37:11 2023

@author: DGAVIDIA
"""

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

pd.options.display.max_colwidth = 200
#%matplotlib inline
#Sample corpus of text documents
corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
          'I love green eggs, ham, sausages and bacon!',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'    
]
labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals', 'weather', 'animals']

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 
                          'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
corpus_df
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
#----------------------------------------------------------------------------*
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
#----------------------------------------------------------------------------*
def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(corpus)
norm_corpus
#----------------------------------------------------------------------------*
import spacy
nlp = spacy.load("en_core_web_sm")
#----------------------------------------------------------------------------*
total_vectors = len(nlp.vocab.vectors)
print('Total word vectors:', total_vectors)
#----------------------------------------------------------------------------*
unique_words = list(set([word for sublist in [doc.split() for doc in norm_corpus] for word in sublist]))
word_glove_vectors = np.array([nlp(word).vector for word in unique_words])
pd.DataFrame(word_glove_vectors, index=unique_words)
#------------------------------------------------------------------------------*
from sklearn.manifold import TSNE
# T-distributed Stochastic Neighbor Embedding.
tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(word_glove_vectors)
labels = unique_words
#------------------------------------------------------------------------------*
plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
#------------------------------------------------------------------------------*    
    
doc_glove_vectors = np.array([nlp(str(doc)).vector for doc in norm_corpus])


from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=0)
km.fit_transform(doc_glove_vectors)
cluster_labels = km.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])

pd.concat([corpus_df, cluster_labels], axis=1)











































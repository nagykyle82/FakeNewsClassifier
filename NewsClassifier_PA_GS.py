# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:21:52 2021

@author: nagyk
"""
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

def preprocessor(text):
    # Remove non-word text
    text = (re.sub('[\W]+', ' ', text.lower()))
    return text
            
# Defines standard and Porter tokenizers for preprocessing
def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# NLTK English stopwords for use in grid search
nltk.download('stopwords')
stop = stopwords.words('english')

# Read in news articles, parse into train/test sets
df = pd.read_csv('news.csv', index_col=0)
y_full = df.label
le = LabelEncoder()
le.fit(y_full)
y_full = le.transform(y_full)
X_full = df.drop(['label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, 
                                                    train_size = 0.75,
                                                    test_size = 0.25,
                                                    random_state=0)
X_train = X_train['text'].apply(preprocessor)
X_test = X_test['text'].apply(preprocessor)
X_train = X_train.values
X_test = X_test.values

# Establish Vectorizer -> Classifier pipeline
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None,
                        analyzer='word')
tfidf_pa = Pipeline([('vect', tfidf),
                     ('clf', PassiveAggressiveClassifier())])

param_grid = [{'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__loss': ['hinge', 'squared_hinge'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__loss': ['hinge', 'squared_hinge'],
               'clf__C': [1.0, 10.0, 100.0]}]
gs_tfidf_pa = GridSearchCV(tfidf_pa, param_grid, scoring='accuracy',
                           cv=5, verbose=1, n_jobs=-1)
gs_tfidf_pa.fit(X_train, y_train)
print('Best parameter set: %s' % gs_tfidf_pa.best_params_)
print('CV Accuracy: %.3f' % gs_tfidf_pa.best_score_)
clf = gs_tfidf_pa.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:07:24 2021

@author: nagyk
"""
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def preprocessor(text):
    # Remove non-word text
    text = (re.sub('[\W]+', ' ', text.lower()))
    return text

def tokenizer(text):
    return text.split()

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
                        analyzer='word', ngram_range=(1, 2), stop_words=stop, 
                        tokenizer=tokenizer)
tfidf_nb = Pipeline([('vect', tfidf),
                     ('clf', MultinomialNB(alpha=0.01))])
'''
# Validation Curve
param_range = np.linspace(0.001,0.01, 10)
train_scores, test_scores = validation_curve(estimator=tfidf_nb,
                                             X=X_train,
                                             y=y_train,
                                             param_name='clf__alpha',
                                             param_range=param_range,
                                             cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o', label='training')
plt.plot(param_range, test_mean, color='red', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(estimator=tfidf_nb,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.linspace(
                                                            0.1, 1.0, 10),
                                                        cv = 5,
                                                        n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', label='training')
plt.plot(train_sizes, test_mean, color='red', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show
'''
tfidf_nb.fit(X_train, y_train)
y_pred = tfidf_nb.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Test Accuracy: %.3f' % tfidf_nb.score(X_test, y_test))
print(confmat)
print(classification_report(y_test, y_pred))

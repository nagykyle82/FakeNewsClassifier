# FakeNewsClassifier
The purpose of this project is to develop a model that will accurately predict whether or not a given news article is 'fake' or 'real'.

The dataset consists of >6000 news articles downloaded from the data-flair website.

TfidfVectorizer was applied to the articles, which yields a 'bag-of-words' which is subsequently tranformed to a normalized inverse document frequency vector.  Two algorithms were tested via grid search: a passive aggressive linear model, as well as a multinomial naive-bayes model.  Models were further tuned via 5-fold cross-validation.  For the passive aggressive model, a C regularization parameter of 1.2, as well as a 'hinge' loss function, yielded best results; for the naive-bayes parameter, a smoothing parameter value of 0.01 yielded best results.  The passive aggressive classifier yielded slightly stronger performance.

Results:
Passive Aggressive: precision=0.94, recall=0.95, f1=0.94, accuracy=0.94
Naive Bayes: precision=0.89, recall=0.97, f1=0.93, accuracy=0.93

import sklearn
import collections
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import MultinomialNB

# https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['class', 'text'])
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.25)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
print(list(count_vect.vocabulary_.items())[0:3])
print(len(count_vect.vocabulary_))

# class to numbers
lab_bin = LabelBinarizer()
y_train_bin = lab_bin.fit_transform(y_train)
y_test_bin = lab_bin.fit_transform(y_test)

# train model
clf = MultinomialNB().fit(X_train_counts, y_train_bin)

# check most important words
importanceCount = collections.Counter()
for word, imp in zip(count_vect.vocabulary_.keys(), clf.coef_[0]):
    importanceCount[word] = imp

print('least important words')
print(importanceCount.most_common()[-10:])

print('most important words')
print(importanceCount.most_common()[0:10])

# test
X_test_counts = count_vect.transform(X_test)
pred = clf.predict(X_test_counts)
print(pred)

print('accuracy: ' + str(average_precision_score(y_test_bin, pred)))

# sanity check
print(clf.predict(count_vect.transform(['win big on this offer'])))
print(clf.predict(count_vect.transform(['hi how are you? shall we meet up soon?'])))
print(clf.predict_proba(count_vect.transform(['hi how are you? shall we meet up soon?'])))

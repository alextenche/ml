import sys
import json
import pickle
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

# dataset https://www.kaggle.com/rmisra/news-category-dataset#News_Category_Dataset_v2.json
categories = []
texts = []

with open('data/News_Category_Dataset_v2.json', 'r') as f:
    for line in f:
        line_json = json.loads(line)
        texts.append(line_json['headline'] + ' ' + line_json['short_description'])
        categories.append(line_json['category'])

df = pd.DataFrame(data={'category': categories, 'text': texts, })
print(df.head())

print(df['category'].value_counts())

df_filtered = df.loc[df['category'].isin(['TECH', 'MONEY'])]
print(df_filtered['category'].value_counts())

# Binarise the category labels
lb = preprocessing.LabelBinarizer()

lb.fit(df_filtered['category'])
df_filtered['category_bin'] = lb.transform(df_filtered['category'])
print(df_filtered.head())

steps = [('vectorise', CountVectorizer()),
         ('transform', TfidfTransformer()),
         ('clf', MultinomialNB())]
pipe = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(df_filtered['text'], df_filtered['category_bin'], test_size=0.25)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print('Accuracy = {:.3f}'.format(f1_score(y_test, pred)))

pickle.dump(pipe, open('models/model.out', 'wb'))

param_grid = dict(vectorise__min_df=[1, 5, 10],
                  vectorise__stop_words=[None,'english'],
                  vectorise__binary=[True,False],
                  # clf__class_weight=['balanced'],
                  transform__norm=['l1','l2'])

grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=make_scorer(f1_score), n_jobs=3)
res = grid_search.fit(df_filtered['text'], df_filtered['category_bin'])
print(res.best_params_)

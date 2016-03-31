import elasticsearch
import elasticsearch.helpers
import sklearn2pmml
import scipy.sparse
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn_pandas
import sklearn.feature_extraction.text
import pandas


data = ["dog cat goldfish", "cat goldfish hamster", "goldfish hamster shark"]
targets = [0, 0, 1]
mapper = sklearn_pandas.DataFrameMapper([
  ('text', sklearn.feature_extraction.text.CountVectorizer()),
], sparse=True)
dataframe=pandas.DataFrame({'text': data})
print mapper.fit_transform(dataframe).toarray()
matrix=mapper.fit_transform(dataframe)
clf = sklearn.linear_model.LogisticRegression()
clf.fit(matrix, targets)
print matrix[2:3]
print(clf.predict(matrix[2:3]))
print(targets[2])
# this does not work because of https://github.com/jpmml/jpmml-sklearn/issues/4

sklearn2pmml.sklearn2pmml(clf, mapper, "simple_example.pmml", with_repr = True)

import elasticsearch
import elasticsearch.helpers
import sklearn2pmml
import scipy.sparse
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn_pandas
import sklearn.feature_extraction.text
import pandas

es = elasticsearch.Elasticsearch()
q= {
 "min_score":0.999,
 "analyzed_text": {
    "analyzer": "standard",
    "field": "text"
  },
  "fields": [
    "label"
  ],
  "query": {
    "function_score": {
      "functions": [
        {"random_score": {}}
      ]
    }
  }
}
# get length 
reg_search = es.search(index='movie-reviews', doc_type='review',body=q)

num_docs = reg_search['hits']['total']

docs = elasticsearch.helpers.scan(es, query=q, scroll=u'10m', index='movie-reviews', doc_type='review')

data = []
targets = []
for doc in docs:
    doc_data = doc['fields']['analyzed_text'][0]
    if (len(doc_data) > 0): 
        text = ''
        for word in doc_data:
            text=text+' '+word
        data.append(text)
        if (doc['fields']['label'][0] =='negative'): 
            targets.append(0)
        else :
            targets.append(1)
    else :
        print 'found empty doc'

print len(data)
mapper4 = sklearn_pandas.DataFrameMapper([
  ('text', sklearn.feature_extraction.text.CountVectorizer()),
], sparse=True)
dataframe=pandas.DataFrame({'text': data})
matrix=mapper4.fit_transform(dataframe)
clf = sklearn.linear_model.LogisticRegression()
clf.fit(matrix, targets)
print matrix[2:3]
print(clf.predict(matrix[2:3]))
print(targets[2])
# this does not work because of https://github.com/jpmml/jpmml-sklearn/issues/4
sklearn2pmml.sklearn2pmml(clf, mapper4, "naive_bayes.pmml", with_repr = True)

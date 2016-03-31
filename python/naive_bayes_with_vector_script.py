import elasticsearch
import elasticsearch.helpers
import sklearn2pmml
import scipy.sparse
import sklearn.naive_bayes
import sklearn.linear_model

es = elasticsearch.Elasticsearch()
q= {
  "script_fields": {
    "vector": {
      "script": {
        "id": "movie_reviews_spec",
        "lang": "pmml_vector"
      }
    }
  },
  "fields": [
    "label"
  ]
}
# get length 
reg_search = es.search(index='movie-reviews', doc_type='review',body=q)

num_docs = reg_search['hits']['total']

docs = elasticsearch.helpers.scan(es, query=q, scroll=u'10m', index='movie-reviews', doc_type='review')
length = reg_search['hits']['hits'][0]['fields']['vector'][0]['length']
i = 0
data = []
indices = []
indicesptr = [0]
indexptr = 0
targets = []
for doc in docs:
    doc_data = doc['fields']['vector'][0]['values']
    doc_indices = doc['fields']['vector'][0]['indices']
    if (len(doc_data) > 0): 
        data.extend(doc_data[:])
        indices.extend(doc_indices[:])
        
        indexptr = len(doc_data) + indexptr
        indicesptr.append(indexptr)
        if (doc['fields']['label'][0] =='negative'): 
            targets.append(0)
        else :
            targets.append(1)
    else :
        print 'found empty doc'

print len(data)
print len(indices)
print len(indicesptr)
matrix = scipy.sparse.csr_matrix((data, indices, indicesptr))
clf = sklearn.linear_model.LogisticRegression()
clf.fit(matrix, targets)
print matrix[2:3]
print(clf.predict(matrix[2:3]))
print(targets[2])

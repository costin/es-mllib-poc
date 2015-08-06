# Howto run the examples

## setup data and elasticsearch

To setup data and elasticsearch and start elasticsearch, run 

```
./example-elasticsearch/setup-example.sh
```

This will 

- download and extract the test data from http://www.cs.cornell.edu/people/pabo/movie-review-data/ (2000 movie reviews taged as "postive" or "negative")
- - download and extract the test data http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip (about 1M tweets taged as "postive" or "negative")
- download and extract elasticsearch 1.5.2
- download make and install [es-token-plugin](from https://github.com/brwe/es-token-plugin)
- install marvel
- start elasticsearch with default configuration and __dynamic scripting enabled__


To start ide of your choice:

```
./gradlew build -x test
./gradlew eclipse idea
```


To index the movie review data run 

```
./gradlew execute -PmainClass=poc.LoadMovieReviews
```

and for the tweet dataset:

```
./gradlew execute -PmainClass=poc.LoadTwitter
```


# Naive Bayes and SVM for sentiment classification

To train classifiers for the movie review dataset run
 
```
./gradlew execute -PmainClass=poc.MovieReviewsClassifier

```

and for the tweet dataset

 
```
./gradlew execute -PmainClass=poc.TweetClassifier

```

This will train a naive bayes model and an SVM and store the parameters back to elasticsearch under

```
GET model/params/id
```
where id can be `naive_bayes_model_params_(tweets|movies)` for the naive bayes models trained with tweet or movie dataset or `svm_model_params_(tweets|movies)`.

To apply the models to new data in elasticsearch you can run a script like this from anywhere where you would normally use scripts:

```
    {
      "script": "naive_bayes_model_stored_parameters_sparse_vectors",
      "lang": "native",
      "params": {
        "field": "text",
        "index": "model",
        "type": "params",
        "id": "naive_bayes_model_params_tweets"
      }
    }
```
For example:

```
GET sentiment140/tweets/_search
{
  "script_fields": {
    "predicted_label": {
      "script": "naive_bayes_model_stored_parameters_sparse_vectors",
      "lang": "native",
      "params": {
        "field": "text",
        "index": "model",
        "type": "params",
        "id": "naive_bayes_model_params_tweets"
      }
    }
  },
  "aggs": {
    "actual": {
      "terms": {
        "field": "label",
        "size": 10
      },
      "aggs": {
        "predicted": {
          "terms": {
            "script": "naive_bayes_model_stored_parameters_sparse_vectors",
            "lang": "native",
            "params": {
              "field": "text",
              "index": "model",
              "type": "params",
              "id": "naive_bayes_model_params_tweets"
            }
          }
        }
      }
    }
  }
}
```



See also [https://gist.github.com/brwe/3cc40f8f3d6e8edc48ac](https://gist.github.com/brwe/3cc40f8f3d6e8edc48ac) for more examples.

 
# Synonyms with word2vec

TODO: Synonym computation seems broken, takes forever to compute the synonyms not sure why.

After setup as described above run the test `SynonymsWithWord2Vec.synonymsInMovieReviews()` which computes synonyms and writes them to a file. 

## Background

Word2vec computes synonyms like so:

#### Input: 

```
[ ["yeah", "it", "has", "spark"],
  ["and", "elasticsearch", "too"],
  ...
]
```

Each row is one field in a "document".
In elasticsearch this translates to the text in a field after its tokens are analyzed:

```
{
	"field": "Yeah! It has Spark!"
},
{
	"field": "And elasticsearch TOO!"
}
```

#### Output:

A model that can be used to find synonyms:

```
JavaRDD<Iterable<String>> corpus;

//get the corpus
HERE BE ELASTICSEARCH

//fit the model
Word2Vec vectorModel = new Word2Vec();
Word2VecModel model = vectorModel.fit(corpus);

// find a synonym
Tuple2<String, Object>[] similarWords = model.findSynonyms("spark", 10);
```

The output can be saved as a synonym file and then used in a synonym filter in a search analyzer.


The input format (analyzed terms in the original order, not sorted) is needed because the algorithm takes into account proximity of words. 
https://code.google.com/p/word2vec/









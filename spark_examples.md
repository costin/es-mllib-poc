# Howto run the examples

## setup data and elasticsearch

To setup data and elasticsearch and start elasticsearch, run 

```
./example-elasticsearch/setup-example.sh
```

This will 

- download and extract the test data from http://www.cs.cornell.edu/people/pabo/movie-review-data/ (2000 movie reviews taged as "postive" or "negative")
- download and extract elasticsearch 1.5.2
- download make and install [es-token-plugin](from https://github.com/brwe/es-token-plugin)
- install marvel
- start elasticsearch

To index the data and run the tests run 

```
./gradlew build -x test
./gradlew eclipse idea
```

and start the ide of your choice.


To index the data run `LoadMovieReviews` to index the data.

# Naive Bayes for sentiment classification

Run the test `NaiveBayesWithScriptFields.movieReviewsNaiveBayes()`. 
This will train a naive bayes model and store it in elasticsearch as a [search template](https://www.elastic.co/guide/en/elasticsearch/reference/master/search-template.html). To use the template for classification run

```
curl -XGET "http://localhost:9200/movie-reviews/_search/template" -d'
{
    "id": "model_script",
    "params" : {
        "field" : "text"
    }
}'
```

You can look at the model with

```
curl -XGET "http://localhost:9200/_search/template/model_script"
```

This will predict a label for each document in the index movie-reviews and return an aggregation of postive and negative labels. 
 
# Synonyms with word2vec

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








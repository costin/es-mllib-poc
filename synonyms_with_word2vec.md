# Synonyms with word2vec


Run the example:

```
# get data

wget http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz 
tar xf review_polarity.tar.gz 

# build elasticsearch 

mkdir elasticsearch
cd elasticsearch
git clone https://github.com/brwe/elasticsearch .
git checkout analyzed-text

mvn clean -DskipTests $2 package
unzip target/releases/elasticsearch-1.4.3-SNAPSHOT.zip -d ./

# start elasticsearch
cd ..
elasticsearch/elasticsearch-1.4.3-SNAPSHOT/bin/elasticsearch 

# index data

python index_movie_reviews.py

```

After that all should be setup to run the test `BasicEsSparkTest.movieReviews()`.
 

# Background

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


The input format (analyzed terms in the original order, not sorted) is needed I think because the algorithm takes into account proximity of words.
TODO: check references at https://code.google.com/p/word2vec/ if this is actualy true

## elasticsearch hack

Because we currently cannot provide the input from elasticsearch I made a hack and pushed it to my repo.
The branch is here:

https://github.com/brwe/elasticsearch/tree/analyzed-text

The branch contains a commit which adds a feature to get the analyzed text in the right format like this:

```

GET wiki/_search
{
  "analyzed_text": [
    {
      "field": "text",
      "idf_threshold": -1
    }
  ]
}
```

```
   "hits": {
      "total": 84,
      "max_score": 1,
      "hits": [
         {
            "_index": "wiki",
            "_type": "page",
            "_id": "93522",
            "_score": 1,
            "fields": {
               "text": [
                  "in",
                  "lakota",
                  "mythology",
                  "wi",
                  "is",
                  "one",
                  "of",
                  "the",
                  "most",
                  "supreme",
                  "gods",
                  "he",
                  "is",
                  "a",
                  "solar",
                  "deity",
                  "and",
                  "is",
                  "associated",
                  "with",
                  "the",
                  "american",
                  "bison",
                  "he",
                  "is",
                  "the",
                  "father",
                  "of",
                  "whope",
                  "anog",
                  "ite",
                  "attempted",
                  "to",
                  "seduce",
                  "wi",
                  "but",
                  "she",
                  "had",
                  "one",
                  "of",
                  "her",
                  "two",
                  "faces",
                  "changed",
                  "into",
                  "an",
                  "ugly",
                  "visage",
                  "as",
                  "punishment"
               ]
            }
         }
```






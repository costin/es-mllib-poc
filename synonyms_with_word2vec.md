# Synonyms with word2vec

To setup data and elasticsearch and start elasticsearch, run 

```
./example-elasticsearch/setup-example.sh
```

Then run `LoadMovieReviews` to index the data.

After that run the test `SynonymsWithWord2Vec.synonymsInMovieReviews()` which computes synonyms and writes them to a file.
 

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


The input format (analyzed terms in the original order, not sorted) is needed because the algorithm takes into account proximity of words. 
https://code.google.com/p/word2vec/

## elasticsearch token plugin

Because elasticsearch currently cannot provide the input from elasticsearch you need to install es-token-plugin as described above.

see doc at https://github.com/brwe/es-token-plugin#analyzed-text-mapper









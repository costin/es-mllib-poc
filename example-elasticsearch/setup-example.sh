#!/bin/bash

# get the test data, see http://www.cs.cornell.edu/People/pabo/movie-review-data/
mkdir data
cd data
wget -nc  http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz 
tar xf review_polarity.tar.gz

wget -nc http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip .
unzip -o trainingandtestdata.zip

# from http://ai.stanford.edu/~amaas/data/sentiment/
wget -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xvf aclImdb_v1.tar.gz

# get the adult income prdiction dataset from http://archive.ics.uci.edu/ml/datasets/Adult
cd ../knime/knime_workspace
mkdir data
cd data
wget -nc http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
cd ../../..


cd example-elasticsearch

elasticsearchversion=2.2.1

# Install elasticsearch 2.2.1 once it is out. we cannot use it now because token plugin must be based on >2.2.0 because of https://github.com/elastic/elasticsearch/pull/16822
wget -nc https://download.elastic.co/elasticsearch/elasticsearch/elasticsearch-2.2.1.tar.gz
tar xvf elasticsearch-2.2.1.tar.gz
# build plugin  

# for now we compile elasticsearch from 2.2 branch 
#mkdir elasticsearch
#cd elasticsearch
#git clone https://github.com/elastic/elasticsearch .
#git fetch
#git reset --hard origin/2.2
#mvn clean -DskipTests $2 package
#mv distribution/tar/target/releases/elasticsearch-$elasticsearchversion.tar.gz ../
#cd ..
tar xf elasticsearch-$elasticsearchversion.tar.gz

mkdir token-plugin
cd token-plugin
git clone https://github.com/brwe/es-token-plugin .
git fetch
git reset --hard origin/master

mvn clean -DskipTests $2 package

currentpath=$(pwd)
../elasticsearch-$elasticsearchversion/bin/plugin remove es-token-plugin
../elasticsearch-$elasticsearchversion/bin/plugin install "file://localhost$currentpath/target/releases/es-token-plugin-$elasticsearchversion.zip"


# enable dynamic scripting
echo 'script.inline: on' >> ../elasticsearch-$elasticsearchversion/config/elasticsearch.yml
echo 'script.indexed: on' >> ../elasticsearch-$elasticsearchversion/config/elasticsearch.yml

# set reloading of file scripts lower
echo 'watcher.interval: "10s"' >> ../elasticsearch-$elasticsearchversion/config/elasticsearch.yml

# should work on laptop also plus I want to watch videos while aggs are computing
echo 'threadpool.search.size: 3' >> ../elasticsearch-$elasticsearchversion/config/elasticsearch.yml

cd ..

# download kibana
wget -nc https://download.elastic.co/kibana/kibana/kibana-4.4.1-linux-x64.tar.gz
tar xf kibana-4.4.1-linux-x64.tar.gz

# install sense
./kibana-4.4.1-linux-x64/bin/kibana plugin --install elastic/sense

# start elasticsearch
elasticsearch-$elasticsearchversion/bin/elasticsearch 


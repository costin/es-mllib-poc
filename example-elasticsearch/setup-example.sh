#!/bin/bash

# get data

wget -nc  http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz 
tar xf review_polarity.tar.gz 

# Install elasticsearch 1.5.2
wget -nc https://download.elastic.co/elasticsearch/elasticsearch/elasticsearch-1.5.2.tar.gz
tar xvf elasticsearch-1.5.2.tar.gz
# build plugin  

mkdir token-plugin
cd token-plugin
git clone https://github.com/brwe/es-token-plugin .

mvn clean -DskipTests $2 package

currentpath=$(pwd)
../elasticsearch-1.5.2/bin/plugin -i token_plugin -u "file://localhost$currentpath/target/releases/es-token-plugin-0.0.0-SNAPSHOT.zip"

# start elasticsearch
cd ..
elasticsearch-1.5.2/bin/elasticsearch 


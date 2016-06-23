#!/bin/bash



cd example-elasticsearch

elasticsearchversion=2.2.1


mkdir token-plugin
cd token-plugin
git clone https://github.com/brwe/es-token-plugin .
git fetch
git reset --hard origin/master

mvn clean -DskipTests $2 package

currentpath=$(pwd)
../elasticsearch-$elasticsearchversion/bin/plugin remove es-token-plugin
../elasticsearch-$elasticsearchversion/bin/plugin install "file://localhost$currentpath/target/releases/es-token-plugin-$elasticsearchversion.zip"




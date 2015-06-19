package poc;

/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.*;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.search.aggregations.bucket.significant.SignificantStringTerms;
import org.elasticsearch.search.aggregations.bucket.significant.SignificantTerms;
import org.elasticsearch.search.aggregations.bucket.significant.heuristics.SignificanceHeuristicBuilder;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import scala.Serializable;
import scala.Tuple2;

import java.util.*;

import static org.elasticsearch.search.aggregations.AggregationBuilders.significantTerms;
import static org.elasticsearch.search.aggregations.AggregationBuilders.terms;
import static scala.collection.JavaConversions.propertiesAsScalaMap;


class ClassifierBase implements Serializable {


    protected static final Properties ES_SPARK_CFG = new Properties();

    static {
        ES_SPARK_CFG.setProperty("es.nodes", "localhost");
        ES_SPARK_CFG.setProperty("es.port", "9200");
    }

    protected static final transient SparkConf conf = new SparkConf().setAll(propertiesAsScalaMap(ES_SPARK_CFG)).setMaster(
            "local[3]").setAppName("estest");

    protected static transient JavaSparkContext sc = null;


    protected void trainClassifiersAndWriteModels(String[] featureTerms, Client client, String indexAndType, String modelSuffix) {
        // get for each document a vector of tfs for the featureTerms
        JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc, indexAndType,
                restRequestBody(featureTerms));

        // convert to labeled point (label + vector)
        JavaRDD<LabeledPoint> corpus = convertToLabeledPoint(esRDD);

        // Split data into training (60%) and test (40%).
        // from https://spark.apache.org/docs/1.2.1/mllib-naive-bayes.html
        JavaRDD<LabeledPoint>[] splits = corpus.randomSplit(new double[]{6, 4});
        JavaRDD<LabeledPoint> training = splits[0];
        JavaRDD<LabeledPoint> test = splits[1];

        // try naive bayes
        System.out.println("train Naive Bayes ");
        training.persist(StorageLevel.MEMORY_AND_DISK());
        final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
        evaluate(test, model);

        // index template search request that can be used for classification of new data
        client.preparePutIndexedScript("mustache", "naive_bayes_model" + modelSuffix, naiveBayesSearchTemplate(model, featureTerms)).get();
        // slow version but with parameters built in
        client.preparePutIndexedScript("groovy", "naive_bayes_model" + modelSuffix, getNaiveBayesGroovyScript(model, featureTerms)).get();

        // try svm
        System.out.println("train SVM ");
        training.persist(StorageLevel.MEMORY_AND_DISK());
        final SVMModel svmModel = SVMWithSGD.train(training.rdd(), 10, 0.1, 0.01, 1);
        evaluate(test, svmModel);

        // index template search request that can be used for classification of new data
        client.preparePutIndexedScript("mustache", "svm_model" + modelSuffix, svmSearchTemplate(svmModel, featureTerms)).get();
        // slow version but with parameters built in
        client.preparePutIndexedScript("groovy", "svm_model" + modelSuffix, getSMVGroovyScript(svmModel, featureTerms)).get();
    }

    private void evaluate(JavaRDD<LabeledPoint> test, final ClassificationModel model) {
        JavaRDD predictionAndLabel = test.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
            public Tuple2<Double, Double> call(LabeledPoint s) {
                return new Tuple2<>(model.predict(s.features()), s.label());
            }
        });
        double accuracy = 1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            public Boolean call(Tuple2<Double, Double> s) {
                return s._1().equals(s._2());
            }
        }).count() / test.count();
        System.out.println("accuracy : " + accuracy);
    }

    private JavaRDD<LabeledPoint> convertToLabeledPoint(JavaPairRDD<String, Map<String, Object>> esRDD) {
        JavaRDD<LabeledPoint> corpus = esRDD.map(
                new Function<Tuple2<String, Map<String, Object>>, LabeledPoint>() {
                    public LabeledPoint call(Tuple2<String, Map<String, Object>> dataPoint) {
                        Double doubleLabel = getLabel(dataPoint);
                        double[] doubleFeatures = getFeatures(dataPoint);
                        return new LabeledPoint(doubleLabel, Vectors.dense(doubleFeatures));
                    }

                    private double[] getFeatures(Tuple2<String, Map<String, Object>> dataPoint) {
                        // convert ArrayList to double[]
                        ArrayList features = (ArrayList) (dataPoint._2().get("vector"));
                        // field values are an array in an array
                        features = (ArrayList) features.get(0);
                        double[] doubleFeatures = new double[features.size()];
                        for (int i = 0; i < features.size(); i++) {
                            doubleFeatures[i] = ((Number) features.get(i)).doubleValue();
                        }
                        return doubleFeatures;
                    }

                    private Double getLabel(Tuple2<String, Map<String, Object>> dataPoint) {
                        // convert string to double label
                        String label = (String) ((ArrayList) dataPoint._2().get("label")).get(0);
                        return label.equals("positive") ? 1.0 : 0;
                    }
                }
        );
        // print some lines so we know how the data looks like
        System.out.println("example document vector: " + corpus.take(1));
        return corpus;
    }

    // request body for vector representation of documents
    private String restRequestBody(String[] featureTerms) {
        return "{\n" +
                "  \"script_fields\": {\n" +
                "    \"vector\": {\n" +
                "      \"script\": \"vector\",\n" +
                "      \"lang\": \"native\",\n" +
                "      \"params\": {\n" +
                "        \"features\": " + Arrays.toString(featureTerms) +
                "        ,\n" +
                "        \"field\": \"text\"\n" +
                "      }\n" +
                "    }\n" +
                "  },\n" +
                "  \"fields\": [\n" +
                "    \"label\"\n" +
                "  ]\n" +
                "}";
    }

    //
    protected String[] getSignificantTermsAsStringList(int numTerms, SignificanceHeuristicBuilder heuristic, Client client, String index) {

        SearchResponse searchResponse = client.prepareSearch(index).addAggregation(terms("classes").field("label").subAggregation(significantTerms("features").field("text").significanceHeuristic(heuristic).size(numTerms / 2))).get();
        List<Terms.Bucket> labelBucket = ((Terms) searchResponse.getAggregations().asMap().get("classes")).getBuckets();
        Collection<SignificantTerms.Bucket> significantTerms = ((SignificantStringTerms) (labelBucket.get(0).getAggregations().asMap().get("features"))).getBuckets();
        List<String> features = new ArrayList<>();
        addFeatures(significantTerms, features);
        significantTerms = ((SignificantStringTerms) (labelBucket.get(1).getAggregations().asMap().get("features"))).getBuckets();
        addFeatures(significantTerms, features);
        String[] featureTerms = features.toArray(new String[features.size()]);
        return featureTerms;
    }

    private void addFeatures(Collection<SignificantTerms.Bucket> significantTerms, List<String> featureArray) {
        for (SignificantTerms.Bucket bucket : significantTerms) {
            featureArray.add("\"" + bucket.getKey() + "\"");
        }
    }

    private String naiveBayesSearchTemplate(NaiveBayesModel model, String[] features) {
        return "{" +
                "  \"template\": {\n" +
                "    \"script_fields\": {\n" +
                "      \"predicted_label\": {\n" +
                "        \"params\": {\n" +
                "          \"features\": " + Arrays.deepToString(features) + ",\n" +
                "          \"field\": \"{{field}}\",\n" +
                "          \"thetas\": " + Arrays.deepToString(model.theta()) + ",\n" +
                "          \"labels\": " + Arrays.toString(model.labels()) + ",\n" +
                "          \"pi\": " + Arrays.toString(model.pi()) + "\n" +
                "        },\n" +
                "        \"script\": \"naive_bayes_model\",\n" +
                "        \"lang\": \"native\"\n" +
                "      }\n" +
                "    },\n" +
                "    \"aggregations\": {\n" +
                "      \"terms\": {\n" +
                "        \"terms\": {\n" +
                "          \"params\": {\n" +
                "            \"features\": " + Arrays.deepToString(features) + ",\n" +
                "            \"field\": \"{{field}}\",\n" +
                "            \"thetas\": " + Arrays.deepToString(model.theta()) + ",\n" +
                "            \"labels\": " + Arrays.toString(model.labels()) + ",\n" +
                "            \"pi\": " + Arrays.toString(model.pi()) + "\n" +
                "          },\n" +
                "          \"script\": \"naive_bayes_model\",\n" +
                "          \"lang\": \"native\"\n" +
                "        }\n" +
                "      }\n" +
                "    },\n" +
                "  \"fields\": [\n" +
                "    \"label\", \"_source\"\n" +
                "  ]\n" +
                "  }\n" +
                "}";
    }

    private String svmSearchTemplate(SVMModel model, String[] features) {
        return "{" +
                "  \"template\": {\n" +
                "    \"script_fields\": {\n" +
                "      \"predicted_label\": {\n" +
                "        \"params\": {\n" +
                "          \"features\": " + Arrays.toString(features) + ",\n" +
                "          \"field\": \"{{field}}\",\n" +
                "          \"weights\": " + Arrays.toString(model.weights().toArray()) + ",\n" +
                "          \"intercept\": " + model.intercept() +
                "        },\n" +
                "        \"script\": \"svm_model\",\n" +
                "        \"lang\": \"native\"\n" +
                "      }\n" +
                "    },\n" +
                "    \"aggregations\": {\n" +
                "      \"terms\": {\n" +
                "        \"terms\": {\n" +
                "          \"params\": {\n" +
                "            \"features\": " + Arrays.toString(features) + ",\n" +
                "            \"field\": \"{{field}}\",\n" +
                "            \"weights\": " + Arrays.toString(model.weights().toArray()) + ",\n" +
                "            \"intercept\": " + model.intercept() +
                "          },\n" +
                "          \"script\": \"svm_model\",\n" +
                "          \"lang\": \"native\"\n" +
                "        }\n" +
                "      }\n" +
                "    },\n" +
                "  \"fields\": [\n" +
                "    \"label\", \"_source\"\n" +
                "  ]\n" +
                "  }\n" +
                "}";
    }

    public String getNaiveBayesGroovyScript(NaiveBayesModel model, String[] features) {
        return "{\"script\": \"" +
                "import org.apache.spark.mllib.classification.NaiveBayesModel;" +
                "import org.elasticsearch.search.lookup.IndexField;" +
                "import org.elasticsearch.search.lookup.IndexFieldTerm;" +
                "import org.apache.spark.mllib.linalg.Vectors;" +
                "features = " + Arrays.toString(features).replace("\"", "\\\"") + ";" +
                "double[] labels = " + Arrays.toString(model.labels()) + ";" +
                "double[][] thetas = " + Arrays.deepToString(model.theta()) + ";" +
                "double[] pi = " + Arrays.toString(model.pi()) + ";" +
                "NaiveBayesModel model = new NaiveBayesModel(labels, pi, thetas);" +
                "double[] tfs = new double[features.size()];" +
                "for (int i = 0; i < features.size(); i++) {" +
                "    tfs[i] = _index[field][features.get(i)].tf();" +
                "}" + ";" +
                "return model.predict(Vectors.dense(tfs));" +
                "\"}";

    }

    public String getSMVGroovyScript(SVMModel model, String[] features) {
        return "{\"script\": \"" +
                "import org.apache.spark.mllib.classification.SVMModel;" +
                "import org.elasticsearch.search.lookup.IndexField;" +
                "import org.elasticsearch.search.lookup.IndexFieldTerm;" +
                "import org.apache.spark.mllib.linalg.Vectors;" +
                "features = " + Arrays.toString(features).replace("\"", "\\\"") + ";" +
                "double[] weights = " + Arrays.toString(model.weights().toArray()) + ";" +
                "double intercept = " + model.intercept() + ";" +
                "SVMModel model = new SVMModel(Vectors.dense(weights), intercept);" +
                "double[] tfs = new double[features.size()];" +
                "for (int i = 0; i < features.size(); i++) {" +
                "    tfs[i] = _index[field][features.get(i)].tf();" +
                "}" + ";" +
                "return model.predict(Vectors.dense(tfs));" +
                "\"}";

    }
}
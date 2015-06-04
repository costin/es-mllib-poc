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
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.node.Node;
import org.elasticsearch.node.NodeBuilder;
import org.elasticsearch.search.aggregations.bucket.significant.SignificantStringTerms;
import org.elasticsearch.search.aggregations.bucket.significant.SignificantTerms;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.elasticsearch.search.aggregations.AggregationBuilders.significantTerms;
import static org.elasticsearch.search.aggregations.AggregationBuilders.terms;
import static scala.collection.JavaConversions.propertiesAsScalaMap;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class NaiveBayesWithScriptFields implements Serializable {

    private static final Properties ES_SPARK_CFG = new Properties();

    static {
        ES_SPARK_CFG.setProperty("es.nodes", "localhost");
        ES_SPARK_CFG.setProperty("es.port", "9200");
    }

    private static final transient SparkConf conf = new SparkConf().setAll(propertiesAsScalaMap(ES_SPARK_CFG)).setMaster(
            "local").setAppName("estest");

    private static transient JavaSparkContext sc = null;

    @BeforeClass
    public static void setup() {
        sc = new JavaSparkContext(conf);
    }

    @AfterClass
    public static void clean() throws Exception {
        if (sc != null) {
            sc.stop();
            // wait for jetty & spark to properly shutdown
            Thread.sleep(TimeUnit.SECONDS.toMillis(2));
        }
    }

    /**
     * This needs token-plugin installed: https://github.com/brwe/es-token-plugin
     */
    @Test
    public void movieReviewsNaiveBayes() throws Exception {
        // use significant terms to get a list of features
        // for example: "bad, worst, ridiculous" for class positive and "awesome, great, wonderful" for class positive
        StringBuilder featureTerms = getSignificantTermsAsStringList(200);
        String target = "movie-reviews/review";
        JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc, target,
                restRequestBody(featureTerms));

        // get the analyzed text from the results
        JavaRDD<LabeledPoint> corpus = esRDD.map(
                new Function<Tuple2<String, Map<String, Object>>, LabeledPoint>() {
                    public LabeledPoint call(Tuple2<String, Map<String, Object>> dataPoint) {
                        Double doubleLabel = getLabel(dataPoint);
                        double[] doubleFeatures = getFeatures(dataPoint);
                        return new LabeledPoint(doubleLabel, Vectors.dense(doubleFeatures));
                    }

                    private double[] getFeatures(Tuple2<String, Map<String, Object>> dataPoint) {
                        // convert ArrayList to double[]
                        ArrayList features = (ArrayList) (dataPoint._2.get("vector"));
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
                        String label = (String) ((ArrayList) dataPoint._2.get("label")).get(0);
                        return label.equals("positive") ? 1.0 : 0;
                    }
                }
        );

        // print some lines so we know how the data looks like
        System.out.println("from esRDD: " + esRDD.take(2));
        System.out.println("from corpus: " + corpus.take(2));
        // Split data into training (60%) and test (40%).
        // from https://spark.apache.org/docs/1.2.1/mllib-naive-bayes.html
        JavaRDD<LabeledPoint>[] splits = corpus.randomSplit(new double[]{1, 2});
        JavaRDD<LabeledPoint> training = splits[0];
        JavaRDD<LabeledPoint> test = splits[1];

        final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
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
        System.out.println("accuracy is " + accuracy);
    }

    private String restRequestBody(StringBuilder featureTerms) {
        return "{\n" +
                "  \"script_fields\": {\n" +
                "    \"vector\": {\n" +
                "      \"script\": \"vector\",\n" +
                "      \"lang\": \"native\",\n" +
                "      \"params\": {\n" +
                "        \"features\": [\n" +
                featureTerms +
                "        ],\n" +
                "        \"field\": \"text\"\n" +
                "      }\n" +
                "    }\n" +
                "  },\n" +
                "  \"fields\": [\n" +
                "    \"label\"\n" +
                "  ]\n" +
                "}";
    }

    private StringBuilder getSignificantTermsAsStringList(int numTerms) {
        StringBuilder featureTerms = new StringBuilder();
        Node node = NodeBuilder.nodeBuilder().client(true).node();
        Client client = node.client();
        SearchResponse searchResponse = client.prepareSearch("movie-reviews").addAggregation(terms("classes").field("label").subAggregation(significantTerms("features").field("text").size(numTerms / 2))).get();
        List<Terms.Bucket> labelBucket = ((Terms) searchResponse.getAggregations().asMap().get("classes")).getBuckets();
        Collection<SignificantTerms.Bucket> significantTerms = ((SignificantStringTerms) (labelBucket.get(0).getAggregations().asMap().get("features"))).getBuckets();
        addFeaturesToString(significantTerms, featureTerms);
        significantTerms = ((SignificantStringTerms) (labelBucket.get(1).getAggregations().asMap().get("features"))).getBuckets();
        addFeaturesToString(significantTerms, featureTerms);
        featureTerms.replace(featureTerms.length() - 1, featureTerms.length(), " ");
        System.out.println("features: " + featureTerms);
        return featureTerms;
    }

    private void addFeaturesToString(Collection<SignificantTerms.Bucket> significantTerms, StringBuilder featureTerms) {
        for (SignificantTerms.Bucket bucket : significantTerms) {
            featureTerms.append("\"");
            featureTerms.append(bucket.getKey());
            featureTerms.append("\"");
            featureTerms.append(",");

        }
    }
}
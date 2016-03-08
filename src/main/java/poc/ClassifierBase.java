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
import org.apache.spark.mllib.pmml.PMMLExportable;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.elasticsearch.common.xcontent.XContentParser;
import org.elasticsearch.common.xcontent.XContentType;
import org.elasticsearch.search.aggregations.bucket.significant.heuristics.SignificanceHeuristicBuilder;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import scala.Serializable;
import scala.Tuple2;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import static org.elasticsearch.common.xcontent.XContentFactory.jsonBuilder;
import static scala.collection.JavaConversions.propertiesAsScalaMap;

/**
 * This needs token-plugin installed: https://github.com/brwe/es-token-plugin
 * <p>
 * Trains a naive bayes classifier and stores the resulting model parameters back to elasticsearch.
 * <p>
 * <p>
 * see https://gist.github.com/brwe/3cc40f8f3d6e8edc48ac for details on how to use
 */
class ClassifierBase implements Serializable {


    protected static final Properties ES_SPARK_CFG = new Properties();

    static {
        ES_SPARK_CFG.setProperty("es.nodes", "localhost");
        ES_SPARK_CFG.setProperty("es.port", "9200");
        ES_SPARK_CFG.setProperty("es.read.unmapped.fields.ignore", "false");
    }

    protected static final transient SparkConf conf = new SparkConf().setAll(propertiesAsScalaMap(ES_SPARK_CFG)).setMaster(
            "local[3]").setAppName("estest");

    protected static transient JavaSparkContext sc = null;


    protected void trainClassifiersAndWriteModels(Map<String, String> featureTerms, Client client, String indexAndType, String modelSuffix) throws IOException {
        // get for each document a vector of tfs for the featureTerms
        JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc, indexAndType,
                restRequestBody(featureTerms));

        // convert to labeled point (label + vector)
        JavaRDD<LabeledPoint> corpus = convertToLabeledPoint(esRDD, Integer.parseInt(featureTerms.get("length")));

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

        System.out.println("write model parameters for naive bayes");
        // index parameters in separate doc

        // try svm
        System.out.println("train SVM ");
        final SVMModel svmModel = SVMWithSGD.train(training.rdd(), 10, 0.1, 0.01, 1);
        evaluate(test, svmModel);

        System.out.println("write model parameters for svm");
        // index parameters in separate doc
        client.prepareIndex("model", "pmml", "svm_model_pmml" + modelSuffix).setSource(getPMMLSource(svmModel, featureTerms)).get();
        //System.out.println(svmModel.toPMML());
        final LogisticRegressionModel lgModel = new LogisticRegressionWithLBFGS()
                .setNumClasses(2)
                .run(training.rdd());
        evaluate(test, lgModel);
        client.prepareIndex("model", "pmml", "lr_model_pmml" + modelSuffix).setSource(getPMMLSource(lgModel, featureTerms)).get();

    }

    private XContentBuilder getPMMLSource(PMMLExportable model, Map<String, String> spec) throws IOException {
        return jsonBuilder().startObject()
                .field("pmml", model.toPMML())
                .field("spec_index", spec.get("spec_index"))
                .field("spec_type", spec.get("spec_type"))
                .field("spec_id", spec.get("spec_id"))
                .endObject();
    }


    private void evaluate(JavaRDD<LabeledPoint> test, final ClassificationModel model) {
        JavaRDD predictionAndLabel = test.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Double> call(LabeledPoint s) {
                return new Tuple2<>(model.predict(s.features()), s.label());
            }
        });
        double accuracy = 1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Double, Double> s) {
                return s._1().equals(s._2());
            }
        }).count() / test.count();
        System.out.println("accuracy for " + model.getClass().getName() + ": " + accuracy);
    }

    private JavaRDD<LabeledPoint> convertToLabeledPoint(JavaPairRDD<String, Map<String, Object>> esRDD, final int vectorLength) {
        JavaRDD<LabeledPoint> corpus = esRDD.map(
                new Function<Tuple2<String, Map<String, Object>>, LabeledPoint>() {
                    @Override
                    public LabeledPoint call(Tuple2<String, Map<String, Object>> dataPoint) {
                        Double doubleLabel = getLabel(dataPoint);
                        return new LabeledPoint(doubleLabel, Vectors.sparse(vectorLength, getIndices(dataPoint), getValues(dataPoint)));
                    }

                    private double[] getValues(Tuple2<String, Map<String, Object>> dataPoint) {
                        // convert ArrayList to double[]
                        Map<String, Object> indicesAndValues = (Map)
                                (((ArrayList) dataPoint._2().get("vector")).get(0));
                        ArrayList valuesList = (ArrayList) indicesAndValues.get("values");
                        if (valuesList == null) {
                            return new double[0];
                        } else {
                            double[] values = new double[valuesList.size()];
                            for (int i = 0; i < valuesList.size(); i++) {
                                values[i] = ((Number) valuesList.get(i)).doubleValue();
                            }
                            return values;
                        }
                    }

                    private int[] getIndices(Tuple2<String, Map<String, Object>> dataPoint) {
                        // convert ArrayList to int[]
                        Map<String, Object> indicesAndValues = (Map) (((ArrayList) dataPoint._2().get("vector")).get(0));
                        ArrayList indicesList = (ArrayList) indicesAndValues.get("indices");
                        if (indicesList == null) {
                            return new int[0];
                        } else {
                            int[] indices = new int[indicesList.size()];
                            for (int i = 0; i < indicesList.size(); i++) {
                                indices[i] = ((Number) indicesList.get(i)).intValue();
                            }
                            return indices;
                        }
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
    private String restRequestBody(Map<String, String> featureTerms) {
        return "{\n" +
                "  \"script_fields\": {\n" +
                "    \"vector\": {\n" +
                "      \"script\": \"vector\",\n" +
                "      \"lang\": \"native\",\n" +
                "      \"params\": {\n" +
                "        \"spec_index\": \"" + featureTerms.get("spec_index") + "\"," +
                "        \"spec_type\": \"" + featureTerms.get("spec_type") + "\"," +
                "        \"spec_id\": \"" + featureTerms.get("spec_id") + "\"" +
                "      }\n" +
                "    }\n" +
                "  },\n" +
                "  \"fields\": [\n" +
                "    \"label\"\n" +
                "  ]\n" +
                "}";
    }

    //
    protected static Map<String, String> prepareAllTermsSpec(String index) {
        String url = "http://localhost:9200/_prepare_spec";
        try {
            URLConnection connection = new URL(url).openConnection();
            connection.setDoOutput(true);
            connection.setDoInput(true);
            connection.setRequestProperty("Accept-Charset", "UTF-8");
            byte[] outputInBytes = getAllTermsSpecRequestBody(index).getBytes("UTF-8");
            OutputStream os = connection.getOutputStream();
            os.write(outputInBytes);
            os.close();
            InputStream is = connection.getInputStream();
            BufferedReader rd = new BufferedReader(new InputStreamReader(is), 1);
            StringBuilder response = new StringBuilder();
            String line;
            while ((line = rd.readLine()) != null) {
                response.append(line);
            }
            rd.close();
            XContentParser parser = XContentFactory.xContent(XContentType.JSON).createParser(response.toString());
            Map<String, Object> params = parser.mapOrdered();
            Map<String, String> finalParams = new HashMap<>();
            finalParams.put("spec_index", (String) params.get("index"));
            finalParams.put("spec_type", (String) params.get("type"));
            finalParams.put("spec_id", (String) params.get("id"));
            finalParams.put("length", Integer.toString((Integer) params.get("length")));
            return finalParams;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        } finally {
        }
    }

    private static String getAllTermsSpecRequestBody(String index) throws IOException {
        XContentBuilder source = jsonBuilder();
        source.startObject()
                .startArray("features")
                .startObject()
                .field("field", "text")
                .field("tokens", "all_terms")
                .field("index", index)
                .field("min_doc_freq", 2)
                .field("number", "occurence")
                .field("type", "string")
                .endObject()
                .endArray()
                .field("sparse", true)
                .endObject();
        return source.string();
    }

    //
    protected Map<String, String> prepareSignificantTermsSpec(int numTerms, SignificanceHeuristicBuilder heuristic, Client client, String index) {
        String url = "http://localhost:9200/_prepare_spec";
        try {
            URLConnection connection = new URL(url).openConnection();
            connection.setDoOutput(true);
            connection.setDoInput(true);
            connection.setRequestProperty("Accept-Charset", "UTF-8");
            byte[] outputInBytes = getSignificantTermsSpecRequestBody(index, numTerms).getBytes("UTF-8");
            OutputStream os = connection.getOutputStream();
            os.write(outputInBytes);
            os.close();
            InputStream is = connection.getInputStream();
            BufferedReader rd = new BufferedReader(new InputStreamReader(is), 1);
            StringBuilder response = new StringBuilder();
            String line;
            while ((line = rd.readLine()) != null) {
                response.append(line);
            }
            rd.close();
            XContentParser parser = XContentFactory.xContent(XContentType.JSON).createParser(response.toString());
            Map<String, Object> params = parser.mapOrdered();
            Map<String, String> finalParams = new HashMap<>();
            finalParams.put("spec_index", (String) params.get("index"));
            finalParams.put("spec_type", (String) params.get("type"));
            finalParams.put("spec_id", (String) params.get("id"));
            finalParams.put("length", Integer.toString((Integer) params.get("length")));
            return finalParams;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        } finally {
        }

    }

    private String getSignificantTermsSpecRequestBody(String index, int numTerms) throws IOException {
        XContentBuilder source = jsonBuilder();
        XContentBuilder request = jsonBuilder();

        request.startObject()
                .startObject("aggregations")
                .startObject("classes")
                .startObject("terms")
                .field("field", "label")
                .endObject()
                .startObject("aggregations")
                .startObject("tokens")
                .startObject("significant_terms")
                .field("field", "text")
                .field("min_doc_count", 0)
                .field("shard_min_doc_count", 0)
                .field("size", numTerms)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject();
        source.startObject()
                .startArray("features")
                .startObject()
                .field("type", "string")
                .field("field", "text")
                .field("tokens", "significant_terms")
                .field("request", request.string())
                .field("index", index)
                .field("number", "occurence")
                .endObject()
                .endArray()
                .field("sparse", true)
                .endObject();
        return source.string();
    }
}
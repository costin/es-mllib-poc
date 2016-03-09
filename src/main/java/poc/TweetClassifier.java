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

import org.apache.spark.api.java.JavaSparkContext;
import org.elasticsearch.client.Client;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.lease.Releasables;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.search.aggregations.bucket.significant.heuristics.JLHScore;

import java.io.IOException;
import java.net.InetAddress;
import java.util.Map;
import java.util.concurrent.TimeUnit;


class TweetClassifier extends ClassifierBase {


    /**
     * This needs token-plugin installed: https://github.com/brwe/es-token-plugin
     * <p/>
     * Trains a naive bayes classifier and stores the resulting model parameters back to elasticsearch.
     * <p/>
     * <p/>
     * see https://gist.github.com/brwe/3cc40f8f3d6e8edc48ac for details on how to use
     */
    public static void main(String[] args) throws IOException {
        Client client = null;
        try {
            sc = new JavaSparkContext(conf);
            client = TransportClient.builder().build()
                    .addTransportAddress(new InetSocketTransportAddress(InetAddress.getByName("localhost"), 9300));
            new TweetClassifier().run(client);
        } finally {
            Releasables.close(client);
            if (sc != null) {
                sc.stop();
                // wait for jetty & spark to properly shutdown
                try {
                    Thread.sleep(TimeUnit.SECONDS.toMillis(2));
                } catch (InterruptedException e) {
                }
            }
        }
    }

    public void run(Client client) throws IOException {
        // use significant terms to get a list of features
        // for example: "bad, worst, ridiculous" for class positive and "awesome, great, wonderful" for class positive
        System.out.println("Get descriptive terms for class positive and negative with significant terms aggregation");
        Map<String, String> spec = prepareSignificantTermsSpec(10000, "sentiment140", "twitter_spec");
        trainClassifiersAndWriteModels(spec, "sentiment140/tweets", "_tweets");
    }
}
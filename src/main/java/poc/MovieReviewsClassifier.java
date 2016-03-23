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

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.TimeUnit;


class MovieReviewsClassifier extends ClassifierBase {

    public static void main(String[] args) throws IOException {
        try {
            sc = new JavaSparkContext(conf);
            new MovieReviewsClassifier().run();
        } finally {
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

    public void run() throws IOException {
        // use significant terms to get a list of features
        // for example: "bad, worst, ridiculous" for class positive and "awesome, great, wonderful" for class positive
        System.out.println("Get descriptive terms for class positive and negative with significant terms aggregation");
        Map<String, String> spec = prepareAllTermsSpec("movie-reviews", "movie_reviews_spec");
        trainClassifiersAndWriteModels(spec, "movie-reviews/review", "_movies");
    }

}
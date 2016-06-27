package poc

import java.net.InetAddress

import org.apache.spark.{SparkConf, SparkContext}
import org.elasticsearch.client.Requests
import org.elasticsearch.client.transport.TransportClient
import org.elasticsearch.common.transport.InetSocketTransportAddress
import org.elasticsearch.index.IndexNotFoundException
import org.elasticsearch.spark.rdd.EsSpark

/**
  * Indexes adult dataset from https://archive.ics.uci.edu/ml/datasets/Adult
  */
object LoadAdult {
  def main(args: Array[String]) = {
    var client = TransportClient.builder().build()
      .addTransportAddress(new InetSocketTransportAddress(InetAddress.getByName("localhost"), 9300));
    println("deleting index adult")
    try {
      client.admin().indices().prepareDelete("adult").get()
    } catch {
      case e: IndexNotFoundException => println("index sentiment140 does not exist")
    }
    client.admin().indices().prepareCreate("adult").addMapping("_default_", "{\n    \"_default_\": {\n      \"dynamic_templates\": [\n        {\n          \"sting_analyzer\": {\n            \"match_mapping_type\": \"string\",\n            \"mapping\": {\n              \"type\": \"string\",\n              \"analyzer\": \"keyword\"\n            }\n          }\n        }\n      ],\n      \"properties\": {\n        \"age\": {\n          \"type\": \"double\"\n        },\n        \"capital_gain\": {\n          \"type\": \"double\"\n        },\n        \"capital_loss\": {\n          \"type\": \"double\"\n        },\n        \"education_num\": {\n          \"type\": \"double\"\n        },\n        \"fnlwgt\": {\n          \"type\": \"double\"\n        },\n        \"hours_per_week\": {\n          \"type\": \"double\"\n        }\n      }\n    } }").get()
    client.admin.cluster.health(Requests.clusterHealthRequest("sentiment140").waitForGreenStatus()).actionGet()
    client.close()
    val path = if (args.length == 1) args(0) else "./data/adult/adult.data"

    new LoadAdult().indexData(path)
  }
}


class LoadAdult extends Serializable {

  @transient lazy val sc = new SparkContext(new SparkConf().setAll(Map("es.nodes" -> "localhost", "es.port" -> "9200")).setMaster("local").setAppName("movie-reviews"))

  def indexData(path: String): Unit = {
    try {
      val csv = sc.textFile(path)
      val rows = csv.map(line => line.split(",").map(_.trim)).map(line =>

        try {
          Opinion(
            try {
              Integer.parseInt(line.apply(0))
            } catch {
              case e: NumberFormatException => null
            },
            if (line.apply(1).equals("?")) {
              ""
            } else {
              line.apply(1)
            },
            try {
              Integer.parseInt(line.apply(2))
            } catch {
              case e: NumberFormatException => null
            },
            line.apply(3),
            try {
              Integer.parseInt(line.apply(4))
            } catch {
              case e: NumberFormatException => null
            },
            line.apply(5),
            if (line.apply(6).equals("?")) {
              ""
            } else {
              line.apply(6)
            },
            line.apply(7),
            line.apply(8),
            line.apply(9),
            try {
              java.lang.Double.parseDouble(line.apply(10))
            } catch {
              case e: NumberFormatException => null
            },
            java.lang.Double.parseDouble(line.apply(11)),
            java.lang.Double.parseDouble(line.apply(12)),
            if (line.apply(13).equals("?")) {
              ""
            } else {
              line.apply(13)
            },
            line.apply(14))
        } catch {
          case e: ArrayIndexOutOfBoundsException => Opinion(null, null, null, null, null, null, null, null, null, null, null, null, null, null, null)
        })
      EsSpark.saveToEs(rows, "adult/train")

    } finally {
      sc.stop()
    }
  }


  case class Opinion(age: Integer, workclass: String, fnlwgt: Integer,
                     education: String, education_num: Integer,
                     marital_status: String, occupation: String, relationship: String,
                     race: String, sex: String, capital_gain: java.lang.Double, capital_loss: java.lang.Double,
                     hours_per_week: java.lang.Double, native_country: String, label: String)

}


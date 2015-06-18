package poc

import org.apache.spark.{SparkConf, SparkContext}
import org.elasticsearch.client.Requests
import org.elasticsearch.indices.IndexMissingException
import org.elasticsearch.node.NodeBuilder
import org.elasticsearch.spark.rdd.EsSpark

/**
 * Indexes tweets into index sentiment140.
 * Documents look like this:
         {
               "label": "negative",
               "text": "\"I am disgustingly full. I hate this feeling! \""
         }
 *
 * Data is from http://help.sentiment140.com/for-students
 */
object LoadTwitter {
  def main(args: Array[String]) = {
    var node = NodeBuilder.nodeBuilder().client(true).node()
    var client = node.client()
    println("deleting index sentiment140")
    try {
      client.admin().indices().prepareDelete("sentiment140").get()
    } catch {
      case e: IndexMissingException => println("index sentiment140 does not exist")
    }
    client.admin().indices().prepareCreate("sentiment140").get()
    client.admin.cluster.health(Requests.clusterHealthRequest("sentiment140").waitForGreenStatus()).actionGet()
    node.close()
    val path = if (args.length == 1) args(0) else "./data/"

    new LoadTwitter().indexData(path)
  }
}

class LoadTwitter extends Serializable {

  @transient lazy val sc = new SparkContext(new SparkConf().setAll(Map("es.nodes" -> "localhost", "es.port" -> "9200")).setMaster("local").setAppName("movie-reviews"))

  def indexData(path: String): Unit = {
    try {
      val csv = sc.textFile(path + "training.1600000.processed.noemoticon.csv")
      val rows = csv.map(line => line.split(",").map(_.trim)).map(line =>
        Opinion(if (line.apply(0).equals("\"4\"")) "positive" else "negative", line.apply(5))
      )
      EsSpark.saveToEs(rows, "sentiment140/tweets")

    } finally {
      sc.stop()
    }
  }


  case class Opinion(label: String, text: String)

}
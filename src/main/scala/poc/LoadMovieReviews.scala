package poc

import java.net.InetAddress

import org.apache.spark.{SparkConf, SparkContext}
import org.elasticsearch.client.Requests
import org.elasticsearch.client.transport.TransportClient
import org.elasticsearch.common.transport.InetSocketTransportAddress
import org.elasticsearch.common.xcontent.json.JsonXContent
import org.elasticsearch.index.IndexNotFoundException
import org.elasticsearch.node.NodeBuilder
import org.elasticsearch.spark.rdd.EsSpark

/**
 * Loads move review data. See http://www.cs.cornell.edu/people/pabo/movie-review-data/ polarity dataset.
 * Extract it under data/ (i.e. project/data/txt_sentoken/neg, pos
 */
object LoadMovieReviews {
  def main(args: Array[String]) = {

    var client = TransportClient.builder().build()
      .addTransportAddress(new InetSocketTransportAddress(InetAddress.getByName("localhost"), 9300));
    println("deleting index movie-reviews")
    try {
      client.admin().indices().prepareDelete("movie-reviews").get()
    } catch {
      case e: IndexNotFoundException => println("index movie-reviews does not exist")
    }
    var mapping = JsonXContent.contentBuilder()
    mapping.startObject()
    mapping.startObject("review")
    mapping.startObject("properties")
    mapping.startObject("text")
    mapping.field("type", "string")
    mapping.field("term_vector", "yes")
    mapping.startObject("fields")
    mapping.startObject("analyzed")
    mapping.field("type", "analyzed_text")
    mapping.field("store", true)
    mapping.endObject()
    mapping.endObject()
    mapping.endObject()
    mapping.endObject()
    mapping.endObject()
    mapping.endObject()
    client.admin().indices().prepareCreate("movie-reviews").addMapping("review", mapping).get()
    client.admin.cluster.health(Requests.clusterHealthRequest("movie-reviews").waitForGreenStatus()).actionGet()
    client.close()
    val path = if (args.length == 1) args(0) else "./data/txt_sentoken"

    new LoadMovieReviews().indexData(path)
    new LoadMovieReviews().indexData("./data/aclImdb/train/")
    new LoadMovieReviews().indexData("./data/aclImdb/test/")
  }
}

class LoadMovieReviews extends Serializable {

  @transient lazy val sc = new SparkContext(new SparkConf().setAll(Map("es.nodes" -> "localhost", "es.port" -> "9200")).setMaster("local").setAppName("movie-reviews"))

  def indexData(path: String): Unit = {
    try {
      // load positive info
      loadFilesInFolder(path + "/pos/", "positive")
      // load negative info
      loadFilesInFolder(path + "/neg/", "negative")

    } finally {
      sc.stop()
    }
  }

  def loadFilesInFolder(path: String, label: String): Unit = {
    val filtered = sc.wholeTextFiles(path).map { tuple =>
      Opinion(label, tuple._2.replace('\n', ' ').replace('"', ' ').replace('\\', ' '))
      // to replace non-Ascii chars add > .replaceAll("[^\\x00-\\x7F]", "")
    }
    EsSpark.saveToEs(filtered, "movie-reviews/review")
  }

  case class Opinion(label: String, text: String)

}
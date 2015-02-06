package poc

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.elasticsearch.spark.rdd.EsSpark

/**
 * Loads move review data. See http://www.cs.cornell.edu/people/pabo/movie-review-data/ polarity dataset.
 * Extract it under data/ (i.e. project/data/txt_sentoken/neg, pos
 */
object LoadMovieReviews {
  def main(args: Array[String]) = {
    val path = if (args.length == 1) args(0) else "./data/txt_sentoken"
    new LoadMovieReviews().indexData(path)
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
    val filtered = sc.wholeTextFiles(path).map{ tuple =>
      Opinion(label, tuple._2.replace('\n',' ').replace('"', ' ').replace('\\',' '))
      // to replace non-Ascii chars add > .replaceAll("[^\\x00-\\x7F]", "")
    }
    EsSpark.saveToEs(filtered, "movie-reviews/review")
  }

  case class Opinion(label: String, text: String)
}
package assinment

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}

import scala.util.control._
import org.apache.spark.ml.clustering.KMeans

import math._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.mutable.ListBuffer

object taks1 {
  val ss = SparkSession.builder().master("local[*]").appName("assigment").getOrCreate()

  import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

  def main(args: Array[String]): Unit = {
    // Create the spark session first
    ss.sparkContext.setLogLevel("ERROR")
    val inputFile = "./Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"

    println("reading from input file: " + inputFile)

    // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
    val basicDF = ss.read.option("header", "true").csv(inputFile)
    val colname = "clean_speech"
    val notnulldf = basicDF.filter(basicDF("member_name").isNotNull && basicDF(colname).isNotNull)

    val Seg = 5
    val udf_segmentTime = udf((v: String) => (v.takeRight(4).toInt - 1989) / Seg)

    val dfwithseg = notnulldf.select($"member_name", $"sitting_date", $"member_name", col(colname), udf_segmentTime($"sitting_date").as("Segment"))

    //time consuming
    val dfwordsTovec = worsdToVec(dfwithseg, colname)

    println("ALG0")
    val loop = new Breaks;
    loop.breakable {
      for (seg <- 0 to 4) {
        val segmenteddf = dfwordsTovec.filter($"Segment" === seg)
        val keywordsofsampleData = algo_topics(segmenteddf, 5)

        println("Time: " + (1989 + seg * 6) + " until " + (1989 + (seg + 1) * 6))
        var ccc = 0
        keywordsofsampleData.foreach(x => {
          println("Cluster " + ccc)
          println()
          x.foreach(println)
          println()
          ccc += 1
        })
      }
    }
    println("Time: " + 1989 + " until " + 2020)
    var ccc = 0
    val keywordsofAll = algo_topics(dfwordsTovec, 5)
    keywordsofAll.foreach(x => {
      println("Cluster " + ccc)
      println()
      x.foreach(println)
      println()
      ccc += 1
    })
  }

  // Add a column of "features" based on column with name "clean_speech"->String
  // based on predefine algorithm
  def worsdToVec(notnulldf: DataFrame, colname: String): DataFrame = {
    val udf_clean = udf((s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", ""))

    val newDF = notnulldf.withColumn("cleaner", udf_clean(col(colname))).persist()

    val tokenizer = new Tokenizer().setInputCol("cleaner").setOutputCol("Words")
    val wordsDF = tokenizer.transform(newDF)

    val word2Vec = new Word2Vec()
      .setInputCol("Words")
      .setOutputCol("result")
      .setVectorSize(50)
      .setMinCount(0)
      .setNumPartitions(10)
    val modelw = word2Vec.fit(wordsDF)
    val resultDf = modelw.transform(wordsDF)

    val featureDf = resultDf.withColumnRenamed("result", "features")
    featureDf
  }

  //Cluster features with k-means
  //calculate tf,idf
  //keep values from sparse vector and calculate keywords (calcKeywords) in each cluster
  def algo_topics(featureDf: DataFrame, number_clusters: Int): List[Array[(String, Double)]] = {


    val kmeans = new KMeans().setK(number_clusters).setSeed(1L).setMaxIter(100)
    val model = kmeans.fit(featureDf)

    val predictions = model.transform(featureDf)
    //======================= OK ========================================

    //==================== TEST ==========================================
    val AllculsterKeyWords = ListBuffer[Array[(String, Double)]]()
    for (clusterk <- 0 until (number_clusters)) {
      val cluster0 = predictions.filter(predictions("prediction") === clusterk)
      // TF-IDF
      val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
      val featurizedDF = hashingTF.transform(cluster0)

      val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
      val idfM = idf.fit(featurizedDF)
      val completeDF = idfM.transform(featurizedDF)

      // method to apply on columns
      val udf_Values_Vector = udf((v: SparseVector) => v.values)
      val udf_Values_TfVector = udf((v: SparseVector) => v.values.map(x => x / v.values.size))

      val complete_valuesDF = completeDF.select($"Words", $"member_name", $"features", $"prediction", udf_Values_TfVector($"rRawFeatures").as("tf_value"), udf_Values_Vector($"rFeatures").as("tf_idf_value"))

      //udf: calculates the distance from center of cluster
      val distance_from_center = udf((features: Vector, c: Int) => sqrt(Vectors.sqdist(features, model.clusterCenters(c))))

      val completeTF_IDF_DF = complete_valuesDF.select($"Words", $"member_name", $"features", $"prediction", $"tf_value", $"tf_idf_value", distance_from_center($"features", $"prediction").as("dist"))
      val completeTF_IDF_DF_Non_Empty = completeTF_IDF_DF.filter(Row => Row.get(0).asInstanceOf[Seq[String]].nonEmpty)
      // Given a group of speeches extract  most representative keywords.
      // method-2
      val keywords = calcKeywords(completeTF_IDF_DF_Non_Empty, 40, 1)
      AllculsterKeyWords.append(keywords)
    }
    AllculsterKeyWords.toList

  }

  // keep most significant words based on tfidf and distance from the center of cluster.
  def calcKeywords(completeTF_IDF_DF: DataFrame, n: Int, k: Int): Array[(String, Double)] = {
    val most_significant_k = udf((Words: List[String], tfidf: Array[Double]) => {
      var sign_words = List[String]()
      for (i <- 0 until min(k, Words.size)) {
        val maxx = tfidf.reduceLeft(_ max _)
        val indexmax = tfidf.indexOf(maxx)
        val wordd = Words(indexmax)
        sign_words = sign_words ::: List(wordd)
        tfidf(indexmax) = -1
      }
      sign_words
    })
    val most_significant_k_tfidf = udf((Words: List[String], tfidf: Array[Double]) => {
      var sign_words = List[String]()
      var sign_tfidf = List[Double]()
      for (i <- 0 until min(k, Words.size)) {
        val maxx = tfidf.reduceLeft(_ max _)
        val indexmax = tfidf.indexOf(maxx)
        val wordd = Words(indexmax)
        sign_words = sign_words ::: List(wordd)
        sign_tfidf = sign_tfidf ::: List(maxx)
        tfidf(indexmax) = -1
      }

      sign_tfidf
    })

    val mswords = completeTF_IDF_DF.select(most_significant_k($"Words", $"tf_idf_value").as("sign_words"), $"dist", most_significant_k_tfidf($"Words", $"tf_idf_value").as("sign_tf_idf"))
    val rdd = mswords.rdd
    val rdd0 = rdd.map(row => (row(0).asInstanceOf[Seq[String]], row(1).asInstanceOf[Double], row(2).asInstanceOf[Seq[Double]])).map(x => (x._1.head, x._3.head / x._2)).reduceByKey(max).map(x => (x._2, x._1))
    rdd0.filter(x => !filterstopwords(x._2)).top(n).map(x => (x._2, x._1))
  }

  def filterstopwords(word: String): Boolean = {
    word.endsWith("ώ") || word.endsWith("ω") || word.endsWith("ει") || word.length <= 3 ||
      word.endsWith("γιατί") || word.endsWith("αλλά") || word.endsWith("ότι") || word.endsWith("αυτό") ||
      word.endsWith("αυτή") || word.endsWith("αυτά") || word.endsWith("εσείς") || word.endsWith("αυτοί") ||
      word.endsWith("καλά") || word.endsWith("εμείς") || word.endsWith("λέτε") || word.endsWith("μόνο") ||
      word.endsWith("οποία") || word.endsWith("αυτήν") || word.endsWith("δεκτό") || word.endsWith("μαι") ||
      word.endsWith("πολύ") || word.endsWith("όλος") || word.endsWith("είναι") || word.endsWith("όχι") ||
      word.endsWith("όπως") || word.endsWith("δύο") || word.endsWith("εάν") || word.endsWith("όμως") ||
      word.endsWith("οποίο") || word.endsWith("όταν") || word.endsWith("όσα") || word.endsWith("τώρα") ||
      word.endsWith("έχουν") || word.endsWith("κύριοι") || word.endsWith("κύριος") || word.endsWith("κυρία") ||
      word.endsWith("θέμα") || word.endsWith("λόγο") || word.endsWith("έτσι") || word.endsWith("ήταν") ||
      word.endsWith("όπου") || word.endsWith("τρία") || word.endsWith("τίποτα") || word.endsWith("υπέρ") ||
      word.endsWith("σήμερα") || word.endsWith("ίδιος") || word.endsWith("ούτε") || word.endsWith("λοιπόν") ||
      word.endsWith("επειδή") || word.endsWith("συνεπώς") || word.endsWith("πώς") || word.endsWith("αυτές") ||
      word.endsWith("αφού") || word.endsWith("ορίστε") || word.endsWith("δηλαδή") || word.endsWith("αρχή") ||
      word.endsWith("έχετε") || word.endsWith("σχετικά") || word.endsWith("λεπτό") || word.endsWith("πρόεδρος") ||
      word.endsWith("υπουργέ") || word.endsWith("συνάδελφος") || word.endsWith("υπουργός")
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

}

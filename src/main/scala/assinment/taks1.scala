package assinment

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control._
import org.apache.spark
import org.apache.spark.ml.clustering.KMeans

import math._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, column, udf}

import scala.collection.mutable.ListBuffer

object taks1 {
  val ss = SparkSession.builder().master("local[*]").appName("assigment").getOrCreate()
  import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

  def main(args: Array[String]): Unit = {
    // Create the spark session first
    ss.sparkContext.setLogLevel("ERROR")
    val timeee = time {
      val inputFile = "./Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"
      //val currentDir = System.getProperty("user.dir")  // get the current directory
      //val outputDir = "file://" + currentDir + "/output"

      println("reading from input file: " + inputFile)

      // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
      val basicDF = ss.read.option("header", "true").csv(inputFile)
      val sampleDF = basicDF.sample(0.01, 1234)
      //sample set
      //val notnulldf = sampleDF.filter(sampleDF("member_name").isNotNull && sampleDF("clean_speech").isNotNull)
      //ALL set
      val notnulldf = sampleDF.filter(sampleDF("member_name").isNotNull && sampleDF("clean_speech").isNotNull)

      //val keywordsofsampleData=algo_topics(notnulldf, ss)
      val Seg = 5
      val udf_segmentTime = udf((v: String) => (v.takeRight(4).toInt - 1989) / Seg)

      val dfwithseg = notnulldf.select($"member_name", $"sitting_date", $"member_name", $"clean_speech", udf_segmentTime($"sitting_date").as("Segment"))

      //time consuming
      val dfwordsTovec = worsdToVec(dfwithseg)

      println("ALG0")
      val loop = new Breaks;
      loop.breakable {
        for (seg <- 0 to 4) {
          val segmenteddf = dfwordsTovec.filter($"Segment" === seg)
          val keywordsofsampleData = algo_topics(segmenteddf,5)

          println("Time: "+(1989+seg*6)+" until "+(1989+(seg+1)*6))
          var ccc=0
          keywordsofsampleData.foreach(x=>{
            println("Cluster "+ccc)
            println()
            x.foreach(println)
            println()
            ccc+=1
          })
        }
      }
    }

    //=====================================================================================================
    //                                How topics change across years
    //=====================================================================================================
    // algo_topics(All data):{
    //   step-1 find cluster
    //   for each cluster find keywords
    // }
    //
    // algo_topics(time segment of data)
  }


  // TODO look for faster/better word2Vec algorithm (tf-idf?)
  // Add a column of "features" based on column with name "clean_speech"->String
  // based on predefine algorithm
  def worsdToVec(notnulldf: DataFrame): DataFrame ={
    val udf_clean = udf((s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", ""))

    val newDF = notnulldf.withColumn("cleaner",udf_clean(col("clean_speech"))).persist()

    val tokenizer = new Tokenizer().setInputCol("cleaner").setOutputCol("Words")
    val wordsDF = tokenizer.transform(newDF)
    /*val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(wordsDF)
    featurizedDF.printSchema()

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)
    completeDF.printSchema()
    */

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
  //keep values from sparse vector and calculate keywords (method2keywords) in each cluster
  def algo_topics(featureDf: DataFrame,number_clusters: Int): List[Array[String]] = {


    val kmeans = new KMeans().setK(number_clusters).setSeed(1L).setMaxIter(100)
    val model = kmeans.fit(featureDf)

    val predictions = model.transform(featureDf)
    //======================= OK ========================================

    //==================== TEST ==========================================
    val AllculsterKeyWords  = ListBuffer[Array[String]]()
    for(clusterk <- 0 until(number_clusters-1)){
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

      val  completeTF_IDF_DF = complete_valuesDF.select($"Words", $"member_name", $"features", $"prediction", $"tf_value", $"tf_idf_value", distance_from_center($"features", $"prediction").as("dist"))
      val completeTF_IDF_DF_Non_Empty = completeTF_IDF_DF.filter(Row => Row.get(0).asInstanceOf[Seq[String]].nonEmpty)
      // Given a group of speeches extract  most representative keywords.
      // method-2
      val keywords = method2keywords(completeTF_IDF_DF_Non_Empty, 40, 1)
      AllculsterKeyWords.append(keywords)
    }
    AllculsterKeyWords.toList

  }
  // method-1 keep k most significant words per speech. Then keep top n frequent words.
  def method1keywords(completeTF_IDF_DF: DataFrame, n: Int, k: Int) {
    val most_significant_k = udf((Words: List[String], tfidf: Array[Double]) => {
      var sign_words = List[String]()
      val k = 40
      for (i <- 0 until min(k, Words.size)) {
        val maxx = tfidf.reduceLeft(_ max _)
        val indexmax = tfidf.indexOf(maxx)
        val wordd = Words(indexmax)
        sign_words = sign_words ::: List(wordd)
        tfidf(indexmax) = -1
      }
      sign_words
    })

    val mswords = completeTF_IDF_DF.select(most_significant_k($"Words", $"tf_idf_value").as("sign_words"))
    val rdd = mswords.rdd
    val rdd0 = rdd.map(_ (0).asInstanceOf[Seq[String]]).flatMap(x => x).map(word => (word, 1)).reduceByKey(_ + _).map(x => (x._2, x._1)) //.sortByKey(false)
    rdd0.filter(x => !filterstopwords(x._2)).top(n).foreach(println)
    rdd0.top(n).map(x => x._2)
  }

  // keep most significant words based on tfidf and distance from the center of cluster.
  def method2keywords(completeTF_IDF_DF: DataFrame, n: Int, k: Int): Array[String] = {
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
    //score=tf_idf/dist
    val rdd0 = rdd.map(row => (row(0).asInstanceOf[Seq[String]], row(1).asInstanceOf[Double], row(2).asInstanceOf[Seq[Double]])).map(x => (x._1.head, x._3.head / x._2)).reduceByKey(max).map(x => (x._2, x._1))

    // Reporte: write about score, which one is better ? (kanena)

    //score=tf_idf
    //val rdd0 = rdd.map(row => (row(0).asInstanceOf[Seq[String]], row(1).asInstanceOf[Double], row(2).asInstanceOf[Seq[Double]])).map(x => (x._1.head, x._3.head)).reduceByKey(max).map(x => (x._2, x._1))
    // val rdd0 = rdd.map(row => (row(0).asInstanceOf[Seq[String]], row(1).asInstanceOf[Double])).map(x => (x._1.head, x._2)).reduceByKey(min).map(x => (x._2, x._1)).sortByKey(true)


    //rdd0.filter(x => !filterstopwords(x._2)).take(n).foreach(println)
    rdd0.filter(x => !filterstopwords(x._2)).top(n).map(x => x._2)
  }
  def filterstopwords(word: String): Boolean = {
    word.endsWith("ώ") || word.endsWith("ω") || word.endsWith("ει") ||
      word.endsWith("γιατί") || word.endsWith("αλλά") || word.endsWith("ότι") || word.endsWith("αυτό") ||
      word.endsWith("αυτή") || word.endsWith("αυτά") || word.endsWith("εσείς") || word.endsWith("αυτοί")
  }
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

}

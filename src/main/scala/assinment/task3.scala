package assinment

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control._
import org.apache.spark
import org.apache.spark.ml.clustering.KMeans

import math._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list, column, concat, concat_ws, udf}
object task3 {
  val ss = SparkSession.builder().master("local[*]").appName("assigment").getOrCreate()
  import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames
  def main(args: Array[String]): Unit = {
    // Create the spark session first
    ss.sparkContext.setLogLevel("ERROR")
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

    val udf_clean = udf((s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", ""))

    val newDF = notnulldf.withColumn("cleaner",udf_clean(col("clean_speech"))).persist()
    //group by member and concat all speeches
    val member_groupDf=newDF.select($"member_name",$"cleaner").groupBy($"member_name").agg(concat_ws(" ",collect_list("cleaner")).as("grouped_speeches"))

    // tokenize
    val tokenizer = new Tokenizer().setInputCol("grouped_speeches").setOutputCol("Words")
    val wordsDF = tokenizer.transform(member_groupDf)

    //TF-EDF
    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(wordsDF)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    //udf to keep only values of TF and IDF vector
    val udf_Values_Vector = udf((v: SparseVector) => v.values)
    val udf_Values_TfVector = udf((v: SparseVector) => v.values.map(x => x / v.values.size))

    val complete_valuesDF = completeDF.select($"Words", $"member_name", $"grouped_speeches",  udf_Values_TfVector($"rRawFeatures").as("tf_value"), udf_Values_Vector($"rFeatures").as("idf_value"))

    val udf_tf_idf = udf((tf: Array[Double], idf: Array[Double]) => {
      for (i <- 0 to idf.length - 1) {
        tf(i) = tf(i) * idf(i)
      }
      tf
    })

    val complete_tf_idf_DF = complete_valuesDF.select($"Words", $"member_name", $"grouped_speeches",$"tf_value",$"idf_value",udf_tf_idf($"tf_value", $"idf_value").as("tf_idf_value"))

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

    val signDf=complete_tf_idf_DF.select($"Words", $"member_name", $"grouped_speeches",$"tf_value",$"idf_value",$"tf_idf_value",most_significant_k($"Words",$"tf_idf_value").as("keywords"))
    val finaldf=signDf.orderBy($"member_name").select($"member_name",$"keywords")

    val udfArrayToString = udf((Words: List[String]) => {
        Words.mkString(",")
      }
    )

  }
}

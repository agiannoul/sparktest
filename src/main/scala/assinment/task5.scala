package assinment

import assinment.taks1.{algo_topics, ss, time, worsdToVec}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control._
import org.apache.spark
import org.apache.spark.ml.clustering.KMeans

import math._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, column, udf}

object task5 {
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

    notnulldf.show()
  }
}

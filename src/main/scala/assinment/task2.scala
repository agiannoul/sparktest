package assinment

import assinment.taks1.worsdToVec
import assinment.task5.ss
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, SparseVector, Vector, Vectors}

import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import scala.util.control._
import math._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{array, col, collect_list, column, concat, concat_ws, lit, udf}
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, IntegerType, StringType, StructField, StructType}

import java.awt.Color
import java.io.File
import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.rdd.RDD
import spire.math.Polynomial.x

object task2 {
  val ss = SparkSession.builder().master("local[*]").appName("assigment").getOrCreate()
  import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

  def main(args: Array[String]): Unit = {

    ss.sparkContext.setLogLevel("ERROR")

    val inputFile = ".././Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"
    println("Task2: reading from input file: " + inputFile)

    // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
    val basicDF = ss.read.option("header", "true").csv(inputFile)
    //sample set
    val sampleDF = basicDF//.sample(0.001, 1234)
    // remove null values from df
    val notnulldf = sampleDF.filter(sampleDF("member_name").isNotNull && sampleDF("clean_speech").isNotNull)

    val distinctValuesDF = notnulldf.select(notnulldf("member_name")).distinct
    //distinctValuesDF.show(5)

    //println(distinctValuesDF.count)
    //val groupByMemberNameMap = notnulldf.groupBy($"member_name")

    //val dfwordsTovec = worsdToVec(notnulldf).filter($"member_name" === "ζιωγας ηλια ιωαννης")//.agg($"features")
    val dfwordsTovec = worsdToVec(notnulldf).groupBy($"member_name").agg(Summarizer.mean($"features").alias("summed_features"))

    //create cosine similarity matrix
    //https://stackoverflow.com/questions/47010126/calculate-cosine-similarity-spark-dataframe
    //https://stackoverflow.com/questions/57530010/spark-scala-cosine-similarity-matrix
    //val temp = convertIndexedRowMatrixToRDD(convertDataFrameToIndexedMatrix(dfwordsTovec))

    val wordsTovecrdd = dfwordsTovec.rdd.map(r=>( r(0).asInstanceOf[String],r(1).asInstanceOf[DenseVector]))
    val cart = wordsTovecrdd.cartesian(wordsTovecrdd)
    val similarityies = cart.filter(u=> u._1._1 != u._2._1).map(x=> (x._1._1 ,x._2._1, cosineSimilarity(x._1._2.values,x._2._2.values)))
    similarityies.take(10).foreach(println)

  }


  //from https://gist.github.com/geekan/471cc0d10f7ecfc769fc
  //==========================================================
  def cosineSimilarity(x: Array[Double], y: Array[Double]): Double = {
    require(x.length == y.length)
    dotProduct(x, y)/(magnitude(x) * magnitude(y))
  }

  /*
   * Return the dot product of the 2 arrays
   * e.g. (a[0]*b[0])+(a[1]*a[2])
   */
  def dotProduct(x: Array[Double], y: Array[Double]): Double = {
    (for((a, b) <- x zip y) yield a * b) sum
  }

  /*
   * Return the magnitude of an array
   * We multiply each element, sum it, then square root the result.
   */
  def magnitude(x: Array[Double]): Double = {
    math.sqrt(x map(i => i*i) sum)
  }



  //============================================================
  def convertDataFrameToIndexedMatrix(df:DataFrame):IndexedRowMatrix = {
    val rows:Long = df.count()
    val cols = df.columns.length
    val rdd = df.rdd.map(
      row => IndexedRow(rows, org.apache.spark.mllib.linalg.Vectors.dense(row.getAs[Seq[Double]](1).toArray)))
    val row = new IndexedRowMatrix(rdd,rows,cols)
    row
  }

  def convertIndexedRowMatrixToRDD(irm:IndexedRowMatrix):RDD[IndexedRow]=irm.rows

}

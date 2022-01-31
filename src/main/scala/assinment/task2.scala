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
  val ss = SparkSession.builder().master("local[*]").config("spark.driver.memory","14g").appName("assigment").getOrCreate()
  import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

  def main(args: Array[String]): Unit = {

    ss.sparkContext.setLogLevel("ERROR")

    val inputFile = "./Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"
    println("Task2: reading from input file: " + inputFile)

    // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
    val basicDF = ss.read.option("header", "true").csv(inputFile)
    //sample set
    val sampleDF = basicDF//.sample(0.1, 1234)
    // remove null values from df
    val notnulldf = sampleDF.filter(sampleDF("member_name").isNotNull && sampleDF("clean_speech").isNotNull)

    idfmethod(notnulldf)
    /*
    //val distinctValuesDF = notnulldf.select(notnulldf("member_name")).distinct
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
    similarityies.map(x=> (x._3,x._1,x._2)).top(10).foreach(println)

     */

  }

  def idfmethod(notnulldf : DataFrame): Unit ={

    val udf_clean = udf((s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", ""))

    val newDF = notnulldf.withColumn("cleaner", udf_clean(col("clean_speech")))
    //val countdf=notnulldf.groupBy("member_name").count().filter($"count" >= 2500)
    //val names97=countdf.select("member_name").collect().map(x=>x.get(0).asInstanceOf[String])
    //println(countdf.count())
    //val isin97 = udf((s: String) => names97.contains(s))

    //val newDF97=newDF.filter(isin97($"member_name"))
    val groupedspeeches=newDF.groupBy($"member_name").agg(concat_ws(" ",collect_list("cleaner")).as("grouped_speeches"))
    val tokenizer = new Tokenizer().setInputCol("grouped_speeches").setOutputCol("Words")
    val wordsDF = tokenizer.transform(groupedspeeches)

    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(wordsDF)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    val membemrfeaturedf =completeDF.select($"member_name",$"rFeatures".as("tf_ifd"))

    val temp=membemrfeaturedf.rdd.map(x=> ( x(0).asInstanceOf[String],x(1).asInstanceOf[SparseVector])).persist()
    //val aggregatedrdd =temp.map(x => (x._1,maxindex(x._2))).persist()
    println(temp.count())

    val cart = temp.cartesian(temp)

    val similarityies = cart.filter(u=> u._1._1.compare(u._2._1)>0).map(x=> (x._1._1 ,x._2._1, sparseSimilarity(x._1._2,x._2._2)))
    //println(similarityies.count())
    similarityies.map(x=> (x._3,x._1,x._2)).top(100).foreach(println)
  }
  // iterable from group reduced to one sparsevector with max in each indice
  def maxindex(vectors : Iterable[SparseVector]): SparseVector = {
    //dot product
    val result = vectors.reduce((x,y)=>{
      val unionindices = x.indices.union(y.indices)
      val sortunionindices = unionindices.distinct.sorted
      val values = sortunionindices.map( i => max(x(i),y(i)))
      val maxvector =new  SparseVector(x.size,sortunionindices,values)
      maxvector
    })
    result
  }
  //cosine similarity of two spaese vector
  def sparseSimilarity(x: SparseVector, y: SparseVector): Double = {
    //dot product
    val common_indicies=x.indices.intersect(y.indices)
    val dotproduct = common_indicies.map(i => x(i) * y(i))
      .foldLeft(implicitly[Numeric[Double]].zero)(_ + _)
    val xmetro= math.sqrt(x.values.map(a => a * a).sum)
    val ymetro= math.sqrt(y.values.map(a => a * a).sum)
    dotproduct/(xmetro*ymetro)
  }


}

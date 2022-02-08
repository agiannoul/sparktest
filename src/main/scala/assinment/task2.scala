package assinment

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector

import math._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list, concat_ws, udf}

object task2 {
  val ss = SparkSession.builder().master("local[*]").config("spark.driver.memory", "14g").appName("assigment").getOrCreate()

  import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

  def main(args: Array[String]): Unit = {

    ss.sparkContext.setLogLevel("ERROR")

    val inputFile = "./Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"
    println("Task2: reading from input file: " + inputFile)

    // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
    val basicDF = ss.read.option("header", "true").csv(inputFile)
    // remove null values from df
    val notnulldf = basicDF.filter(basicDF("member_name").isNotNull && basicDF("clean_speech").isNotNull)

    idfmethod(notnulldf)
  }

  def idfmethod(notnulldf: DataFrame): Unit = {

    val udf_clean = udf((s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", ""))

    val newDF = notnulldf.withColumn("cleaner", udf_clean(col("clean_speech")))
    val groupedspeeches = newDF.groupBy($"member_name").agg(concat_ws(" ", collect_list("cleaner")).as("grouped_speeches"))
    val tokenizer = new Tokenizer().setInputCol("grouped_speeches").setOutputCol("Words")
    val wordsDF = tokenizer.transform(groupedspeeches)

    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures")
    val featurizedDF = hashingTF.transform(wordsDF)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    val membemrfeaturedf = completeDF.select($"member_name", $"rFeatures".as("tf_ifd"))

    val temp = membemrfeaturedf.rdd.map(x => (x(0).asInstanceOf[String], x(1).asInstanceOf[SparseVector])).persist()
    println(temp.count())

    val cart = temp.cartesian(temp)

    val similarityies = cart.filter(u => u._1._1.compare(u._2._1) > 0).map(x => (x._1._1, x._2._1, sparseSimilarity(x._1._2, x._2._2)))
    similarityies.map(x => (x._3, x._1, x._2)).top(100).foreach(println)
  }

  // iterable from group reduced to one sparsevector with max in each indice
  def maxindex(vectors: Iterable[SparseVector]): SparseVector = {
    //dot product
    val result = vectors.reduce((x, y) => {
      val unionindices = x.indices.union(y.indices)
      val sortunionindices = unionindices.distinct.sorted
      val values = sortunionindices.map(i => max(x(i), y(i)))
      val maxvector = new SparseVector(x.size, sortunionindices, values)
      maxvector
    })
    result
  }

  //cosine similarity of two spaese vector
  def sparseSimilarity(x: SparseVector, y: SparseVector): Double = {
    //dot product
    val common_indicies = x.indices.intersect(y.indices)
    val dotproduct = common_indicies.map(i => x(i) * y(i))
      .foldLeft(implicitly[Numeric[Double]].zero)(_ + _)
    val xmetro = math.sqrt(x.values.map(a => a * a).sum)
    val ymetro = math.sqrt(y.values.map(a => a * a).sum)
    dotproduct / (xmetro * ymetro)
  }
}

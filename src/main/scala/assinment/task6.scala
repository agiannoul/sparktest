package assinment

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

import scala.math.{Numeric, max}

object task6 {
  val ss = SparkSession.builder().master("local[*]").appName("assigment").getOrCreate()

  import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames
  def main(args: Array[String]): Unit = {
    // Create the spark session first
    ss.sparkContext.setLogLevel("ERROR")
    val inputFile = "./Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"

    println("reading from input file: " + inputFile)

    // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
    val basicDF = ss.read.option("header", "true").csv(inputFile)
    val notnulldf = basicDF.filter(basicDF("member_name").isNotNull && basicDF("clean_speech").isNotNull)
    val udf_clean = udf((s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", ""))

    val newDF = notnulldf.withColumn("cleaner", udf_clean(col("speech")))

    val tokenizer = new Tokenizer().setInputCol("cleaner").setOutputCol("Words")
    val wordsDF = tokenizer.transform(newDF)

    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures")
    val featurizedDF = hashingTF.transform(wordsDF)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    val tempcompletedf = completeDF.select($"member_name", $"political_party", $"rFeatures".as("tf_ifd"))
    val partyfeaturedf = tempcompletedf.select($"political_party", $"tf_ifd")

    val temp = partyfeaturedf.rdd.map(x => (x(0).asInstanceOf[String], x(1).asInstanceOf[SparseVector])).groupByKey()
    val aggregatedrdd = temp.map(x => (x._1, maxindex(x._2)))

    val membemrfeaturedf = tempcompletedf.select($"member_name", $"tf_ifd").filter(checkforspecific($"member_name"))
    val temp2 = membemrfeaturedf.rdd.map(x => (x(0).asInstanceOf[String], x(1).asInstanceOf[SparseVector])).groupByKey()
    val aggregatedrdd2 = temp2.map(x => (x._1, maxindex(x._2)))
    val cart = aggregatedrdd2.cartesian(aggregatedrdd)
    val similarityies = cart.map(x => (x._1._1, x._2._1, sparseSimilarity(x._1._2, x._2._2)))
    similarityies.map(x => (x._3, x._1, x._2)).foreach(println)
  }


  def checkforspecific(coln: Column): Column = {
    (
      (coln.contains("????????????????????") && coln.contains("??????????????????????"))
        ||
        (coln.contains("??????????????") && coln.contains("??????????????"))
        ||
        (coln.contains("????????????") && coln.contains("??????????????????"))
        ||
        (coln.contains("????????????????") && coln.contains("??????????"))
      )
  }

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

  //cosine similarity of two sparse vector
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

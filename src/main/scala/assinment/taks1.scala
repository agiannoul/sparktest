package assinment

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.{SparseVector, Vectors, Vector}
import org.apache.spark.{SparkConf, SparkContext}

import math._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, column, udf}

object taks1 {
  def main(args: Array[String]): Unit = {
    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("assigment").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
    import ss.implicits._  // For implicit conversions like converting RDDs to DataFrames

    val inputFile = "./Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"
    //val currentDir = System.getProperty("user.dir")  // get the current directory
    //val outputDir = "file://" + currentDir + "/output"

    println("reading from input file: " + inputFile)

    // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
    val basicDF =  ss.read.option("header", "true").csv(inputFile)
    val sampleDF= basicDF.sample(0.01,1234)


    val notnulldf=sampleDF.filter(basicDF("member_name").isNotNull && basicDF("clean_speech").isNotNull)
    val udf_clean = udf( (s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", "") )

    val newDF = notnulldf.select($"member_name", udf_clean($"clean_speech").as("cleaner")).persist()

    val tokenizer = new Tokenizer().setInputCol("cleaner").setOutputCol("Words")
    val wordsDF = tokenizer.transform(newDF)

    //wordsDF.show(5)
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
    val modelw = word2Vec.fit(wordsDF)
    val resultDf=modelw.transform(wordsDF)


    import org.apache.spark.ml.clustering.KMeans

    val featureDf=resultDf.withColumnRenamed("result","features")
    val kmeans = new KMeans().setK(5).setSeed(1L).setMaxIter(100)
    val model = kmeans.fit(featureDf)

    val predictions = model.transform(featureDf)
    model.clusterCenters.foreach(println)
    //======================= OK ========================================

    //==================== TEST ==========================================
    val cluster0 = predictions.filter(predictions("prediction") === 0 )

    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(cluster0)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)


    val udf_Values_Vector = udf( (v: SparseVector) => v.values )
    val udf_Values_TfVector = udf( (v: SparseVector) => v.values.map(x=> x/v.values.size) )

    val complete_valuesDF=completeDF.select($"Words",$"member_name",$"features",$"prediction",udf_Values_TfVector($"rRawFeatures").as("tf_value"),udf_Values_Vector($"rFeatures").as("idf_value"))
    val udf_tf_idf = udf( (tf: Array[Double],idf: Array[Double]) => {
      for (i <- 0 to idf.length-1){
          tf(i)=tf(i)*idf(i)
      }
      tf
    } )
    val distance_from_center =  udf((features: Vector, c: Int) => sqrt(Vectors.sqdist(features, model.clusterCenters(c))))

    val completeTF_IDF_DF=complete_valuesDF.select($"Words",$"member_name",$"features",$"prediction",$"tf_value",$"idf_value",udf_tf_idf($"tf_value",$"idf_value").as("tf_idf_value"),distance_from_center($"features",$"prediction").as("rank"))
    completeTF_IDF_DF.show(5)
  }

}
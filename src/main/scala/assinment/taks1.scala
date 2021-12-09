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
    // TF-IDF
    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(cluster0)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    // method to apply on columns
    val udf_Values_Vector = udf( (v: SparseVector) => v.values )
    val udf_Values_TfVector = udf( (v: SparseVector) => v.values.map(x=> x/v.values.size) )

    val complete_valuesDF=completeDF.select($"Words",$"member_name",$"features",$"prediction",udf_Values_TfVector($"rRawFeatures").as("tf_value"),udf_Values_Vector($"rFeatures").as("idf_value"))
    // udf to combine tf and idf values
    val udf_tf_idf = udf( (tf: Array[Double],idf: Array[Double]) => {
      for (i <- 0 to idf.length-1){
          tf(i)=tf(i)*idf(i)
      }
      tf
    } )
    //udf: calculates the distance from center of cluster
    val distance_from_center =  udf((features: Vector, c: Int) => sqrt(Vectors.sqdist(features, model.clusterCenters(c))))

    val completeTF_IDF_DF=complete_valuesDF.select($"Words",$"member_name",$"features",$"prediction",$"tf_value",$"idf_value",udf_tf_idf($"tf_value",$"idf_value").as("tf_idf_value"),distance_from_center($"features",$"prediction").as("dist"))
    completeTF_IDF_DF.show(5)

    // Given a group of speeches extract  most representative keywords.

    // method-1 keep k most significant words per speech. Then keep top n frequent words.

    val most_significant_k =udf((Words: List[String],tfidf: Array[Double]) => {
      var sign_words = List[String]()
      val k=5
      for( i <-0 until min(k,Words.size)){
        val maxx=tfidf.reduceLeft(_ max _)
        val indexmax=tfidf.indexOf(maxx)
        val wordd=Words(indexmax)
        sign_words =  sign_words ::: List(wordd)
        tfidf(indexmax) = -1
      }
      sign_words
    })
    val mswords = completeTF_IDF_DF.select(most_significant_k($"Words",$"idf_value").as("sign_words"))
    mswords.show(10,false)
    val rdd =mswords.rdd
    val rdd0=rdd.map(_(0).asInstanceOf[Seq[String]]).flatMap(x=>x).map(word => (word,1)).reduceByKey(_+_).sortBy(_._1,false)
    rdd0.top(5).foreach(println)
  }

}

package assinment

import org.apache.spark.ml.clustering.{BisectingKMeans, LDA}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, monotonicallyIncreasingId, monotonically_increasing_id, udf}

import scala.collection.mutable.ListBuffer

object Trends {
  val ss = SparkSession.builder().master("local[*]").appName("assigment").config("spark.driver.memory","16g").getOrCreate()
  import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

  def iter_to_ArrayFreq(iter: Iterable[(Int, Int)]) :Array[Int] = {
    var temp: Array[Int] = new Array[Int](20)
    temp.foreach(x=>0)
    for(tupl <- iter){
      temp(tupl._1)=tupl._2
    }
    temp
  }

  def main(args: Array[String]): Unit = {
    // Create the spark session first
    ss.sparkContext.setLogLevel("ERROR")
    val inputFile = "./Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"
    //val currentDir = System.getProperty("user.dir")  // get the current directory
    //val outputDir = "file://" + currentDir + "/output"

    println("reading from input file: " + inputFile)

    // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
    val basicDF = ss.read.option("header", "true").csv(inputFile)
    val sampleDF = basicDF.sample(0.5, 1234)
    //sample set

    val notnulldf = sampleDF.filter(sampleDF("member_name").isNotNull && sampleDF("clean_speech").isNotNull)
    val udf_clean = udf((s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", ""))

    val newDF = notnulldf.withColumn("cleaner", udf_clean(col("speech")))

    val tokenizer = new Tokenizer().setInputCol("cleaner").setOutputCol("Words")
    val wordsDF = tokenizer.transform(newDF)
    val udf_wordclean = udf((s: Seq[String]) => {
      var a :ListBuffer[String] = ListBuffer()
      s.foreach(x=>{

        if(x.length>3 && !filterstopwords(x)){
          a.append(x.replace(".",""))
        }
      })
      a.toSeq
    })
    val df = wordsDF.withColumn("id",monotonically_increasing_id()).select($"id",$"sitting_date",udf_wordclean($"Words").as("Words_clean"))
    val removeEmpty = udf((array: Seq[String]) => !array.isEmpty)

    val df2 = df.filter(removeEmpty($"Words_clean"))

//    val hashingTF = new HashingTF().setInputCol("Words_clean").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
//    val featurizedDF = hashingTF.transform(df2)
//
//    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("features")
//    val idfM = idf.fit(featurizedDF)
//    val completeDF = idfM.transform(featurizedDF).drop("rRawFeatures")


    val vectorizer  : CountVectorizerModel = new CountVectorizer().setInputCol("Words_clean").setOutputCol("features").fit(df2)
    val completeDF= vectorizer.transform(df2)
    //completeDF.show(20)
    val lda = new LDA().setK(20).setMaxIter(20).setFeaturesCol("features")
    val model = lda.fit(completeDF)

    model.describeTopics(200).rdd.map(x=>x(1).asInstanceOf[Seq[Int]]).map(x=>x.map(index=>vectorizer.vocabulary(index))).coalesce(1).saveAsTextFile("./tmp/task6Topics/")

    val transformed = model.transform(completeDF)

    val udf_max_topic=udf((v :DenseVector)=>({
      v.values.indexOf(v.values.max)
    }))
    //===================topics by date====================================
    ///*
    val trendsDf = transformed.withColumn("topic_id",udf_max_topic($"topicDistribution")).select("sitting_date","topic_id")
    val trendrdd = trendsDf.rdd.map(r=>((r(0).asInstanceOf[String],r(1).asInstanceOf[Int]),1)).reduceByKey(_+_).map(r=>(r._1._1,(r._1._2,r._2))) //((date,topic_id),count)
    val trend_date = trendrdd.groupByKey().map(r=> (r._1,iter_to_ArrayFreq(r._2)))
    trend_date.map(r=>""+r._1+","+r._2.mkString(",")).coalesce(1).saveAsTextFile("./tmp/task6/")

     /*/

    //=====================outlier detection====================================
    val topicsDf = transformed.withColumn("topic_id",udf_max_topic($"topicDistribution")).select("topic_id","Words_clean","features")
    val typedrdd=topicsDf.rdd.map(r=>(r(0).asInstanceOf[Int],(r(1).asInstanceOf[Seq[String]],r(2).asInstanceOf[Vector[Double]])))
    for(topicc <- 0 to 19){
      typedrdd.sample(false,0.1)
    }

    */

  }
  def filterstopwords(word: String): Boolean = {
    val stops : Array[String] = Array("άρθρο", "δεκτό", "έχει", "έγινε", "συνεπώς", "κατά", "σώμαγίνεται", "ερωτάται", "τροποποιήθηκε", "κύριο", "υπουργό", "νομοσχέδιο",
      "πρέπει", "στην", "τους","υπάρχουν", "υπάρχει", "έχουμε", "στον", "στις", "προς", "αφορά", "μπορεί", "μέσα", "κυβέρνηση", "κάνει", "γίνει", "είχε",
      "όχι", "νομος", "ιωάννης", "γεώργιος", "κωνσταντίνος", "πολιτική", "χώρα", "κάθε", "δημήτριος", "κύριε", "μετά", "νικόλαος", "στους,", "αθανάσιος", "ένας")
      stops.contains(word) || word.contains("υπουργ") || word.startsWith("αριθμ")|| word.contains("κύριε") || word.contains("πρόεδρε") ||
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
      word.endsWith("έχετε") || word.endsWith("σχετικά") || word.endsWith("λεπτό") || word.isEmpty
  }
}

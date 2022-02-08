package assinment

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}

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
object task3 {
  val ss = SparkSession.builder().master("local[8]").appName("assigment").config("spark.driver.memory","6g").config("spark.executor.memory","2g").getOrCreate()
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
    val sampleDF = basicDF//.sample(0.05, 1234)
    //sample set
    //val notnulldf = sampleDF.filter(sampleDF("member_name").isNotNull && sampleDF("clean_speech").isNotNull)
    //ALL set
    val notnulldf = sampleDF.filter(sampleDF("member_name").isNotNull && sampleDF("clean_speech").isNotNull)

    val udf_clean = udf((s: String) => s.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", ""))

    val newDF = notnulldf.withColumn("cleaner",udf_clean(col("clean_speech")))

    //political_party,member_name
    val Seg = 1 //1 d-> for each year
    val udf_segmentTime = udf((v: String) => (v.takeRight(4).toInt - 1989) / Seg)
    val dfwithseg = newDF.select($"member_name", $"political_party", $"cleaner", udf_segmentTime($"sitting_date").as("Segment"))
    val colname="member_name"

    //analysis for each year

    //create empty dataframe
    val schema = new StructType()
      .add(colname, StringType)
      .add("keywords", ArrayType(StringType))
      .add("keywords_TFIDF", ArrayType(DoubleType))
      .add("Segment", IntegerType)


    var dfAllyears = ss.createDataFrame(ss.sparkContext
      .emptyRDD[Row],schema)
    val loop = new Breaks;
    dfwithseg.cache()
    var firstTime:Boolean= true
    loop.breakable {
      for (seg <- 0 to 30) {
        println(seg)
        val segmenteddf = dfwithseg.filter($"Segment" === seg)
        if(!segmenteddf.isEmpty) {
          val tempdf = groupedAnalysis(segmenteddf, colname, 50)
          val dfwithsegment = tempdf.withColumn("Segment", lit(seg))
          dfAllyears =dfAllyears.union(dfwithsegment)
        }
      }

    }
    dfwithseg.unpersist()

    //dfAllyears df has colums:  col(colunName),$"keywords",$"keywords_TFIDF","Segment"
    val allrdd=dfAllyears.map(row => (row(0).asInstanceOf[String],row(1).asInstanceOf[Seq[String]],row(2).asInstanceOf[Seq[Double]],row(3).asInstanceOf[Int]))
    val partiesrdd  = allrdd.rdd.groupBy(_._1).map(r=>(r._1,dotmatrix(r._2)))


    partiesrdd.collect().foreach(x=> {
      imagecreation(x._2._1,x._1.replaceAll(" ","_"))
    })




    //groupedAnalysis(newDF,"political_party",50)
    //signleAnalysis(newDF,"political_party",50)


  }

  case class Members(name: String, keywords: Seq[String],keywords_TFIDF: Seq[Double],seg: Int)


  def imagecreation(yourmatrix: Array[Array[Double]],name:String){
    val width = yourmatrix.length
    val height = yourmatrix(0).length
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    try {

      var i = 0
      while ( {
        i < height
      }) {
        var j = 0
        while ( {
          j < width
        }) {
          val u = yourmatrix(i)(j)*(255.0/yourmatrix(i).max)
          val myWhite :Color = new Color(u.toInt, u.toInt, u.toInt); // Color white
          image.setRGB(i,j,myWhite.getRGB)

          j += 1
        }

        i += 1
      }
      ImageIO.write(image, "jpg", new File("./tmp/task3members/"+name+"_image.jpg"))

    } catch {
      case e: Exception =>
      e.printStackTrace()
    }
  }


  def dotmatrix(segmentunroder :Iterable[(String,Seq[String],Seq[Double],Int)] ): (Array[Array[Double]],List[Int]) ={
    val dotMatrix= Array.ofDim[Double](segmentunroder.size, segmentunroder.size)
    val segmentIndex =ListBuffer[Int]()
    var countseg=0
    val segment = segmentunroder.toArray.sortBy(x=> x._4)
    for(seg <- segment){
      var countInseg=0
      for(inseg <-segment){
        var dotproduct : Double = 0
        val iteratedWords= ListBuffer[String]()
        for(word <- seg._2){

          val Inindex= inseg._2.indexOf(word)
          if(Inindex != -1 && !iteratedWords.contains(word)) {
            iteratedWords.append(word)
            //println(" index:"+Inindex+" idfs:"+inseg._3.size+" words:"+inseg._2.size)
            val Segindex = seg._2.indexOf(word)
            //dotproduct += inseg._3(Inindex) * seg._3(Segindex)
            //dotproduct += max(inseg._3(Inindex) , seg._3(Segindex))
            dotproduct +=1
          }
        }
        dotMatrix(countseg)(countInseg)=dotproduct
        countInseg+=1
      }
      segmentIndex.append(seg._4)
      countseg+=1
    }
    (dotMatrix,segmentIndex.toList)
  }

  // Tf-IDf in speeches of single member/party
  // take top N words
  def analysisOfspecificmemberDf(name :String,specific_member_df: DataFrame,N:Int): (String,Array[String],Array[Double])={


    val tokenizer = new Tokenizer().setInputCol("cleaner").setOutputCol("Words")
    val wordsDF = tokenizer.transform(specific_member_df)

    //TF-IDF
    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(wordsDF)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    //udf to keep only values of TF and IDF vector
    val udf_Values_Vector = udf((v: SparseVector) => v.values)

    val complete_valuesDF = completeDF.select($"Words", udf_Values_Vector($"rFeatures").as("tf_idf_value"))




    // df -> key(word) , value(tf*idf) rdd
    val rdd = complete_valuesDF.rdd
    val rdd0 = rdd.map(row => (row(0).asInstanceOf[Seq[String]], row(1).asInstanceOf[Seq[String]])).map( x => {
      var concated = ListBuffer[String]()
      for (i <- 0 until min(x._1.size,x._2.size)) {
        concated.append(x._1(i)+" _ "+x._2(i))
      }
      concated
    }).flatMap(x=>x).map(x=> (x.split(" _ ")(0), x.split(" _ ")(1).toDouble )).reduceByKey(max).map(x => (x._2, x._1))
    val top40=rdd0.top(min(N,rdd0.count()).asInstanceOf[Int])
    val top_words=top40.map(x=>x._2)
    val top_tfidf=top40.map(x=>x._1)
    (name,top_words,top_tfidf)

  }
  // filter df based on names of members/parties
  def signleAnalysis(newDF :DataFrame, colunName1 : String,N :Int):Any ={
    val colunName=colunName1
    val distingsmember=newDF.select(col(colunName)).rdd.map(row => row(0).asInstanceOf[String]).distinct()
    val member_keyword=distingsmember.collect().map(name=>{
      val specific_member_df=newDF.filter(col(colunName).equalTo(name))
      val tirple=analysisOfspecificmemberDf(name,specific_member_df,N)
      tirple
    })

    //print
    /*
    member_keyword.take(10).foreach(x=>{
      print(x._1+" ")
      x._2.foreach(a=>print(a+","))
      print(" | ")
      x._3.foreach(a=>print(a+","))
      println()
    })
    */
  }





  // Group by member/party and concat speeches, then calculate tf-idf and select top 40/k
  // return df with columns col(colunName),$"keywords",$"keywords_TFIDF"
  def groupedAnalysis(newDF :DataFrame, colunName1 : String,N :Int): DataFrame={
    //group by member and concat all speeches
    val colunName :String =colunName1
    val member_groupDf=newDF.select(col(colunName),$"cleaner").groupBy(col(colunName)).agg(concat_ws(" ",collect_list("cleaner")).as("grouped_speeches"))

    // tokenize
    val tokenizer = new Tokenizer().setInputCol("grouped_speeches").setOutputCol("Words")
    val wordsDF = tokenizer.transform(member_groupDf)

    //TF-IDF
    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(wordsDF)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    //udf to keep only values of TF and IDF vector
    val udf_Values_Vector = udf((v: SparseVector) => v.values)

    val complete_valuesDF = completeDF.select($"Words", col(colunName), $"grouped_speeches", udf_Values_Vector($"rFeatures").as("tf_idf_value"))

    val udf_tf_idf = udf((tf: Array[Double], idf: Array[Double]) => {
      for (i <- 0 to idf.length - 1) {
        tf(i) = tf(i) * idf(i)
      }
      tf
    })


    val most_significant_k = udf((Words: List[String], tfidf: Array[Double]) => {
      var sign_words = List[String]()
      val k = N
      for (i <- 0 until min(k, Words.size)) {
        val maxx = tfidf.reduceLeft(_ max _)
        if(maxx != -1) {
          val indexmax = tfidf.indexOf(maxx)
          val wordd = Words(indexmax)
          sign_words = sign_words ::: List(wordd)
          tfidf(indexmax) = -1
        }
      }
      sign_words
    })

    val most_significant_k_tfidf = udf((Words: List[String], tfidf: Array[Double]) => {
      var sign_words = List[String]()
      var sign_tfidf = List[Double]()
      val k=40
      for (i <- 0 until min(k, Words.size)) {
        val maxx = tfidf.reduceLeft(_ max _)
        if(maxx != -1) {
          val indexmax = tfidf.indexOf(maxx)
          val wordd = Words(indexmax)
          sign_words = sign_words ::: List(wordd)
          sign_tfidf = sign_tfidf ::: List(maxx)
          tfidf(indexmax) = -1
        }
      }

      sign_tfidf


    })

    val signDf=complete_valuesDF.select($"Words",col(colunName), $"grouped_speeches",$"tf_idf_value",most_significant_k($"Words",$"tf_idf_value").as("keywords"),most_significant_k_tfidf($"Words",$"tf_idf_value").as("keywords_TFIDF"))
    val finaldf=signDf.orderBy(col(colunName)).select(col(colunName),$"keywords",$"keywords_TFIDF")
    //finaldf.show(false)
    finaldf
  }
}

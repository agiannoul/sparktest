package assinment

import assinment.taks1.{filterstopwords, ss, time, worsdToVec}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.{SparseVector}
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.{col, udf}

import scala.math.{min}
import scala.util.control.Breaks

object task4 {

  def main(args: Array[String]): Unit = {

    LogManager.getRootLogger.setLevel(Level.ERROR)
    LogManager.getLogger("org").setLevel(Level.ERROR)
    LogManager.getLogger("akka").setLevel(Level.ERROR)
    ss.sparkContext.setLogLevel("ERROR")
    val inputFile = "./Greek_Parliament_Proceedings_1989_2020_Clean_V2.csv"

    println("reading from input file: " + inputFile)

    // Read the contents of the csv file in a dataframe. The csv file does not contain a header.
    val basicDF = ss.read.option("header", "true").csv(inputFile)
    val sampleDF = basicDF.sample(0.01, 1234)
    //sample set
//    val notnulldf = sampleDF.filter(sampleDF("member_name").isNotNull && sampleDF("clean_speech").isNotNull)
    //ALL set
    val notnulldf = basicDF.filter(basicDF("member_name").isNotNull && basicDF("clean_speech").isNotNull)

    notnulldf.persist()
    val timeee = time {

      val beforeCrisisYears = udf((v: String) => v.takeRight(4).toInt >= 2000 && v.takeRight(4).toInt < 2009)
      val beforeCrisisDF = notnulldf.filter(beforeCrisisYears(col("sitting_date"))).select("member_name", "political_party", "clean_speech")

      // set 2017 as upper bound as after this year the crisis was not the hottest topic
      val afterCrisisYears = udf((v: String) => v.takeRight(4).toInt >= 2009 && v.takeRight(4).toInt <= 2017)
      val afterCrisisDF = notnulldf.filter(afterCrisisYears(col("sitting_date"))).select("member_name", "political_party", "clean_speech")

//      val existingPartiesBefore = beforeCrisisDF.select("political_party").distinct().collect()
//        .toList.map(x => x.mkString)
//      val existingPartiesAfter = afterCrisisDF.select("political_party").distinct().collect()
//        .toList.map(x => x.mkString)
//
//      val politicalLeaders = beforeCrisisDF.select("member_name").distinct()
//        .filter(x => checkForLeader(x(0).toString)).collect().toList.map(x=> x.mkString)
//      val existingParties = existingPartiesBefore.intersect(existingPartiesAfter)
//
//      val finalSpeechesBefore = beforeCrisisDF
//        .filter(x => existingParties.exists(x(1).toString.contains(_)))
//
//      val finalSpeechesAfter = afterCrisisDF
//        .filter(x => existingParties.exists(x(1).toString.contains(_)))

      val dfwordsTovecAfter = worsdToVec(afterCrisisDF.filter(x => checkForParty(x(1).toString)),"clean_speech")
      val dfwordsTovecBefore = worsdToVec(beforeCrisisDF.filter(x => checkForParty(x(1).toString)),"clean_speech")

      val existingParties = List("συνασπισμος ριζοσπαστικης αριστερας", "νεα δημοκρατια",
        "κομμουνιστικο κομμα ελλαδας", "πανελληνιο σοσιαλιστικο κινημα", "εξωκοινοβουλευτικός",
        "ανεξαρτητοι (εκτος κομματος)", "λαικος ορθοδοξος συναγερμος")
      val politicalLeaders = List("καραμανλης αλεξανδρου κωνσταντινος", "σαμαρας κωνσταντινου αντωνιος",
      "παπαρηγα νικολαου αλεξανδρα", "βενιζελος βασιλειου ευαγγελος", "παπανδρεου ανδρεα γεωργιος")

      for (party <- existingParties) {
        val partyAfterDF = dfwordsTovecAfter.filter(x => x(1).toString.equals(party))
        val partyBeforeDF = dfwordsTovecBefore.filter(x => x(1).toString.equals(party))
        println(party)
        println("before")
        val before = calcKeywords(partyBeforeDF)
        println(before.mkString(", "))
        println("after")
        val after = calcKeywords(partyAfterDF)
        println(after.mkString(", "))
        println("diff")
        println(before.union(after).distinct.filter(s => !before.intersect(after).contains(s)).mkString(", "))
      }

      println("\n\n\n")
      for (leader <- politicalLeaders) {
        val partyAfterDF = dfwordsTovecAfter.filter(x => x(0).toString.equals(leader))
        val partyBeforeDF = dfwordsTovecBefore.filter(x => x(0).toString.equals(leader))
        val before = calcKeywords(partyBeforeDF)
        val after = calcKeywords(partyAfterDF)
        val diff = before.union(after).distinct.filter(s => !before.intersect(after).contains(s))

        println(leader)
        println("before")
        println(before.mkString(", "))
        println("after")
        println(after.mkString(", "))
        println("diff")
        println(diff.mkString(", "))
      }
    }
    println(timeee)
  }

  def checkForLeader(member: String): Boolean = {
    ((member.contains("αλεξανδρα") && member.contains("παπαρηγα"))
      ||
      (member.contains("παπανδρεου") && member.contains("γεωργιος"))
      ||
      (member.contains("βενιζελος") && member.contains("ευαγγελος"))
      ||
      (member.contains("κωνσταντινος") && member.contains("καραμανλης"))
      ||
      (member.contains("αντωνιος") && member.contains("σαμαρας"))
      ||
      (member.contains("αλεξιος") && member.contains("τσιπρας")))
  }

  def checkForParty(party: String): Boolean = {
    (party.contains("συνασπισμος ριζοσπαστικης αριστερας")
      ||
      party.contains("νεα δημοκρατια")
      ||
      party.contains("κομμουνιστικο κομμα ελλαδας")
      ||
      party.contains("πανελληνιο σοσιαλιστικο κινημα")
      ||
      party.contains("εξωκοινοβουλευτικός")
      ||
      party.contains("ανεξαρτητοι (εκτος κομματος)")
      ||
      party.contains("λαικος ορθοδοξος συναγερμος"))

  }

  def calcKeywords(party: DataFrame): Array[String] = {
    // TF-IDF
    val hashingTF = new HashingTF().setInputCol("Words").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(party)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    // method to apply on columns
    val udf_Values_Vector = udf((v: SparseVector) => v.values)
    val udf_Values_TfVector = udf((v: SparseVector) => v.values.map(x => x / v.values.size))

    val complete_valuesDF = completeDF.select(col("Words"), col("member_name"),
      col("features"), udf_Values_TfVector(col("rRawFeatures")).as("tf_value"),
      udf_Values_Vector(col("rFeatures")).as("tf_idf_value"))

    // Given a group of speeches extract  most representative keywords.
    // method-1
    method1keywords(complete_valuesDF, 40, 20)
  }

  def method1keywords(completeTF_IDF_DF: DataFrame, n: Int, k: Int): Array[String] = {
    val most_significant_k = udf((Words: List[String], tfidf: Array[Double]) => {
      var sign_words = List[String]()
      for (i <- 0 until min(k, Words.size)) {
        val maxx = tfidf.reduceLeft(_ max _)
        val indexmax = tfidf.indexOf(maxx)
        val wordd = Words(indexmax)
        sign_words = sign_words ::: List(wordd)
        tfidf(indexmax) = -1
      }
      sign_words
    })

    val mswords = completeTF_IDF_DF.select(most_significant_k(col("Words"), col("tf_idf_value")).as("sign_words"))
    val rdd = mswords.rdd
    val rdd0 = rdd.map(_ (0).asInstanceOf[Seq[String]]).flatMap(x => x).map(word => (word, 1)).reduceByKey(_ + _).map(x => (x._2, x._1)) //.sortByKey(false)
    rdd0.filter(x => !filterstopwords(x._2)).top(n).map(x => x._2)
  }


  def checkforparty(coln: Column, parties: List[String]): Boolean = {
    parties.exists(coln.toString.contains(_))
  }
}

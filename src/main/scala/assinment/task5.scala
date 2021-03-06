package assinment

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

object task5 {
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

    val newDF = notnulldf.withColumn("cleaner", udf_clean(col("speech"))).persist()

    val tokenizer = new Tokenizer().setInputCol("cleaner").setOutputCol("Words")
    val wordsDF = tokenizer.transform(newDF)

    val udf_env = udf((s: Seq[String]) => {
      var belong = false
      val enviroment = List("περιβάλλον", "φύση", "απόβλητα", "ρύπανση", "βιολογικός", "ανανεώσιμες", "πηγές", "ενέργει", "κλίμα", "κλιματικ", "περιβαλλον")
      for (e <- enviroment) {
        if (s.contains(e)) {
          belong = true
        }
      }
      belong
    })

    val udf_econ = udf((s: Seq[String]) => {
      var belong = false
      val economy = List("χρέος", "φόρος", "οικονομία", "ανάπτυξη", "ανεργία", "εργασία", "δάνειο", "επίδομα", "μνημόνιο", "λιτότητα", "οικονομ", "τράπεζ", "φορο", "ανεργ", "επίδομα")
      for (e <- economy) {
        if (s.contains(e)) {
          belong = true
        }
      }
      belong
    })

    val udf_def = udf((s: Seq[String]) => {
      var belong = false
      val defence = List("εθνική", "τουρκία", "γείτονα", "άμυνα", "κυριαρχικά", "δικαιώματα", "σύνορα", "μεταναστευτικό", "πρεσπών", "πρόσφυγ", "προσφυγ", "στρατι", "στρατός", "όπλο", "ένοπλ", "αμύνης")
      for (e <- defence) {
        if (s.contains(e)) {
          belong = true
        }
      }
      belong
    })

    val udf_health = udf((s: Seq[String]) => {
      var belong = false
      val health = List("υγεία", "νοσοκομεί", "ιατρό", "κλινικ", "εμβόλι", "φάρμακ", "φαρμακ", "εοδυ", "υγει", "ιατρ", "γρίπη")
      for (e <- health) {
        if (s.contains(e)) {
          belong = true
        }
      }
      belong
    })

    statisticsColumn("political_party", wordsDF.select($"member_name", $"political_party", $"Words"))
    val env_cluster = wordsDF.select($"member_name", $"political_party", $"Words").filter(udf_env($"Words")).persist()
    val eco_cluster = wordsDF.select($"member_name", $"political_party", $"Words").filter(udf_econ($"Words")).persist()
    val def_cluster = wordsDF.select($"member_name", $"political_party", $"Words").filter(udf_def($"Words")).persist()
    val health_cluster = wordsDF.select($"member_name", $"political_party", $"Words").filter(udf_health($"Words")).persist()

    statisticsColumn("political_party", env_cluster)
    statisticsColumn("political_party", eco_cluster)
    statisticsColumn("political_party", def_cluster)
    statisticsColumn("political_party", health_cluster)

    statisticsColumnScaled("political_party", env_cluster, wordsDF)
    statisticsColumnScaled("political_party", eco_cluster, wordsDF)
    statisticsColumnScaled("political_party", def_cluster, wordsDF)
    statisticsColumnScaled("political_party", health_cluster, wordsDF)
  }

  // Participation in df
  def statisticsColumn(colname: String, df: DataFrame): Any = {
    val all = 1.0 * df.count()
    val grouped = df.groupBy(colname).count().as("count")
    val partdf = grouped.withColumn("participation", $"count" / all)
    val n: Int = partdf.count().asInstanceOf[Int]
    partdf.orderBy($"participation".desc).show(n, false)
  }

  //  participation divided by participation in all Dataset
  def statisticsColumnScaled(colname: String, df: DataFrame, wordsdf: DataFrame): Any = {
    val all = 1.0 * df.count()
    val all2 = 1.0 * wordsdf.count()
    val grouped = df.groupBy(colname).count().as("count")
    val thematicpart = grouped.withColumn("participation", $"count" / all)

    val grouped2 = wordsdf.groupBy(colname).count().as("count")
    val generalpart = grouped2.withColumn("participation", $"count" / all2)
    val array = generalpart.select(colname, "participation").collect().map(r => (r(0).asInstanceOf[String], r(1).asInstanceOf[Double]))

    val udf_scaledpart = udf((s: String, p: Double) => {
      var newpart: Double = 0.0
      array.foreach(t => {
        if (t._1.equals(s)) {
          newpart = p / t._2
        }
      })
      newpart
    })
    val n: Int = thematicpart.count().asInstanceOf[Int]
    thematicpart.select(col(colname), udf_scaledpart(col(colname), $"participation").as("Scaled_participation")).orderBy($"Scaled_participation".desc).show(n, false)
  }
}

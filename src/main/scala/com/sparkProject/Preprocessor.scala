package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{RowFactory, SparkSession}
import org.apache.spark.sql.types._

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP
    // on vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation de la SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc et donc aux mécanismes de distribution des calculs.)
    val spark = SparkSession
      .builder
      .config(conf)
      .master("local")
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._


    /** *****************************************************************************
      *
      * TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    println("hello world ! from Preprocessor")

    val trainFilename = Preprocessor.getClass.getResource("/train_clean.csv").getFile
    val cleanedTrainFilename = trainFilename + "_cleaned.csv"
    spark.sparkContext.textFile(trainFilename).map(
      l => l.replaceAll("[\"]{2,}", "").trim
    ).coalesce(1).saveAsTextFile(cleanedTrainFilename)

    val df = spark.read
      .option("header", "true")
      .option("delimiter", ",")
      .csv(cleanedTrainFilename).cache

    // Afficher le nombre de lignes et le nombre de colonnes dans le dataFrame.
    println(s"Dataset has ${df.count} lines")

    // Afficher un extrait du dataFrame sous forme de tableau.
    df.show

    // Afficher le schéma du dataFrame (nom des colonnes et le type des données contenues dans chacune d’elles).
    df.printSchema

    // Assigner le type “Int” aux colonnes qui vous semblent contenir des entiers.
    val df2 = df.withColumn("goal", $"goal".cast(IntegerType))
      .withColumn("backers_count", $"backers_count".cast(IntegerType))
      .withColumn("final_status", $"final_status".cast(IntegerType))

    /** ********
      * Cleaning
      * **********/

    // Afficher une description statistique des colonnes de type Int (avec .describe().show )
    df2.select($"goal", $"backers_count", $"final_status").describe().show()

    // Observer les autres colonnes, et proposer des cleanings à faire sur les données:
    //   - faites des groupBy count, des show, des dropDuplicates.
    //   - Quels cleaning faire pour chaque colonne ?
    //   - Y a-t-il des colonnes inutiles ?
    //   - Comment traiter les valeurs manquantes ?
    //   - Des “fuites du futur” ???

    // project_id
    df2.select($"project_id").distinct.count == df2.select($"project_id").count // True

    df2.groupBy($"project_id").count.filter($"count" > 1).show
    df2.filter(not($"project_id" rlike "kkst*")).count // 161

    // name
    df2.select(trim(lower($"name"))).count
    df2.withColumn("name", trim(lower($"name"))).groupBy($"name").count
      .filter($"count" > 1).count // 340

    // keywords is same as name
    df2.select(regexp_replace(lower($"name"), " ", "-"), $"keywords").show(false)

    // disable_communication|country|currency
    // disable_communication: 311 false and 102171 true
    df2.groupBy("disable_communication").count().describe().show

    // country
    df2.filter(length($"country") === 3).groupBy("country").count().show
    // currency
    df2.groupBy("currency").count().show

    // campaign length may be interested
    df2.select(round(($"deadline" - $"launched_at") / 3600).cast(IntegerType).as("campaign_length")).show()

    val malformed_df2 = df2.filter($"goal" isNull)
      .withColumn("final_status", $"backers_count")
      .withColumn("backers_count", $"launched_at")
      .withColumn("launched_at", $"created_at")
      .withColumn("created_at", $"state_changed_at")
      .withColumn("state_changed_at", $"deadline")
      .withColumn("deadline", $"currency")
      .withColumn("currency", $"country")
      .withColumn("country", $"disable_communication")
      .withColumn("disable_communication", $"keywords")
      .withColumn("keywords", $"goal")
      .withColumn("goal", $"desc")
      .withColumn("desc", $"name")
      .withColumn("name", $"project_id")

    df2.filter($"disable_communication" contains   "mr-squiggles").show


    // project id is not usefull for prediction
    val df3 = df2.filter($"project_id" rlike "kkst*")
      .select($"keywords", $"goal", $"country", $"currency", $"final_status",
        round(($"deadline" - $"launched_at") / 3600).cast(IntegerType).as("campaign_length"))

    val schema = StructType(Array(
      StructField("final_status", IntegerType, true),
      StructField("backers_count", IntegerType, true),
      StructField("launched_at", LongType, true),
      StructField("created_at", LongType, true),
      StructField("state_changed_at", LongType, true),
      StructField("deadline", LongType, true),
      StructField("currency", StringType, true),
      StructField("country", StringType, true),
      StructField("disable_communication", StringType, true),
      StructField("keywords", StringType, true)
    ))

    val header = spark.sparkContext.textFile(cleanedTrainFilename).take(1).head
    val rawData = spark.sparkContext.textFile(cleanedTrainFilename).filter(l => l != header)
      .map(
      l => {
        val fields = l.split(",").reverse
        val rowFields = schema.zipWithIndex.map(
          t => {
            val index = t._2
            val fieldType = t._1
            val value:String = fields(index)
            val rowVal:Any = try{
              fieldType.dataType match {
                case IntegerType => value.toInt
                case DoubleType => value.toDouble
                case LongType => value.toLong
                case _ => value
              }
            } catch {
              case e: Exception => null
            }
            rowVal.asInstanceOf[Object]
          }
        )
        RowFactory.create(rowFields:_*)
      }
    )

    val df5 = spark.createDataFrame(rawData, schema)
    df5.show

    df5.filter(not(length($"country") === 2) and ($"country" rlike "[A-Z]") ).count
    df5.filter(not(length($"currency") === 3) and ($"currency" rlike "[A-Z]") ).count


  }

}

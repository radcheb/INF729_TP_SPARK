package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory


object Trainer {

  private val logger = LoggerFactory.getLogger(Trainer.getClass)

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      logger.error("You need to provide the URI of input dataset and output model params")
      System.exit(1)
    }

    val inputUri = args(0)
    val modelUri = args(1)

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .master("local")
      .appName("TP_spark")
      .getOrCreate()


    /** *****************************************************************************
      *
      * TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    println("hello world ! from Trainer")

    val dataDf = spark.read.parquet(inputUri).withColumnRenamed("final_status", "label")

    // Build pipleline

    // text columns tokenizer
    val textTokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("words")

    val countVectorizer = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("count_vectors")

    val idf = new IDF().setInputCol("count_vectors").setOutputCol("tfidf")

    val countryStringIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country2_indexed")

    val currencyStringIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency2_indexed")

    // OneHotEncoder deprecated in Spark 2.3
    val encoders = new OneHotEncoderEstimator()
      .setInputCols(Array("country2_indexed", "currency2_indexed"))
      .setOutputCols(Array("country2_vect", "currency2_vect"))

    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country2_vect", "currency2_vect")).
      setOutputCol("features")

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStandardization(true)
      .setPredictionCol("prediction")
      .setRawPredictionCol("raw_prediction")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    val stages = Array(textTokenizer, stopWordsRemover, countVectorizer, idf, currencyStringIndexer, countryStringIndexer, encoders, vectorAssembler, lr)

    val pipeline = new Pipeline()
      .setStages(stages)

    // split data to training and test
    val splits = dataDf.randomSplit(Array(0.9, 0.1), seed = 1234)
    val (training, test) = (splits(0).cache, splits(1).cache)

    // Prepare grid search
    val regParams = (-2 to(-8, -2)).toList.map(Math.pow(10, _))
    val minDFs = (55D to(95D, 20)).toList

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, regParams)
      .addGrid(countVectorizer.minDF, minDFs)
      .build()

    // Prepare estimator
    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    // Train model
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(f1Evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setParallelism(2)
    val model = trainValidationSplit.fit(training)

    model.transform(test)
      .select("features", "label", "prediction")
      .show()

    // Evalute best model
    val dfWithPredictions = model.transform(test)
    val bestF1Score = f1Evaluator.evaluate(dfWithPredictions)
    println(s"Best F1 score = $bestF1Score")

    dfWithPredictions.groupBy("label", "prediction").count.show

    // Save Model
    model.write.overwrite().save(modelUri)
  }
}

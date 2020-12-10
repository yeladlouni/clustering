import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object Scoring extends App {
  val spark = SparkSession
    .builder()
    .config("AppName", "Scoring")
    .config("spark.master", "local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val df_train = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv("C:\\Users\\yelad\\IdeaProjects\\logistic-regression\\data\\train.csv")

  val df_validation = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv("C:\\Users\\yelad\\IdeaProjects\\logistic-regression\\data\\validation.csv")

  val df_test = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv("C:\\Users\\yelad\\IdeaProjects\\logistic-regression\\data\\test.csv")


  var houseIndexer = new StringIndexer()
    .setInputCol("has_house")
    .setOutputCol("has_house_indexed")

  val statusIndexer = new StringIndexer()
    .setInputCol("marital_status")
    .setOutputCol("marital_status_indexed")

  val assembler = new VectorAssembler()
    .setInputCols(Array("age", "salary", "has_house_indexed"))
    .setOutputCol("features")

  val model = new KMeans()
    .setK(3)

  val pipeline = new Pipeline()
    .setStages(Array(houseIndexer, statusIndexer, assembler, model))

  pipeline.fit(df_train).transform(df_train).show()



}
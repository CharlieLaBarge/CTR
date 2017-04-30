import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql
import org.apache.spark.sql.Dataset

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


object CTR {
	def main(args: Array[String]) {
        // store dataset locations
        val training_full_location = "/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/training.txt"
        val training_reduced_location = "/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/training_reduced.txt"
        val training_refined_location = "/Users/JCFL/Code/CT/Data Science/Assignment2+3/tester/*.csv"

        // initialize contexts
        val conf = new SparkConf().setAppName("CTR")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")

        // initiate spark session
        val spark = SparkSession.builder()
                                .appName("CTR")
                                .getOrCreate()

        // // read in train data
        // var train_data = spark.read.format("com.databricks.spark.csv") // read in main training file
        //                                 .option("delimiter", "\t")
        //                                 .load(training_reduced_location)
		//
		// var df = train_data
		// // rename all columns, cast certain columns to int
		// df = df.withColumn("Click", df("_c0").cast("Int")).drop("_c0")
		// 					.withColumn("Impression", df("_c1").cast("Int")).drop("_c1")
		// 					.withColumn("DisplayURL", df("_c2")).drop("_c2")
		// 					.withColumn("AdID", df("_c3")).drop("_c3")
		// 					.withColumn("AdvertiserID", df("_c4")).drop("_c4")
		// 					.withColumn("Depth", df("_c5").cast("Int")).drop("_c5")
		// 					.withColumn("Position", df("_c6").cast("Int")).drop("_c6")
		// 					.withColumn("QueryID", df("_c7")).drop("_c7")
		// 					.withColumn("KeywordID", df("_c8")).drop("_c8")
		// 					.withColumn("TitleID", df("_c9")).drop("_c9")
		// 					.withColumn("DescriptionID", df("_c10")).drop("_c10")
		// 					.withColumn("UserID", df("_c11")).drop("_c11")
		//
		// train_data = df
		//
		// // get groupings by AdID and QueryID
		// var groupedByAdAndQuery = df.groupBy("AdID", "QueryID").count().sort(desc("count"))
		//
		// var top_N_instances = groupedByAdAndQuery.limit(25000) // take top 25k instances
		//
		// // prepare for join
		// top_N_instances = top_N_instances.withColumn("AID", top_N_instances("AdID")).drop("AdID")
		// top_N_instances = top_N_instances.withColumn("QID", top_N_instances("QueryID")).drop("QueryID")
		//
		// // inner join based on matching AdIDs and QueryIDs, intended to filter out non
		// var train_refined = df.join(top_N_instances, (df("AdID") <=> top_N_instances("AID")) && (df("QueryID") <=> top_N_instances("QID")))
		//
		// train_data = train_refined
		//
		// // import additional data files
		// var descriptions_df = spark.read.format("com.databricks.spark.csv") // read in descriptions file
		// 	.option("delimiter", "\t")
		// 	.load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/descriptionid_tokensid.txt")
		//
		// var keywords_df = spark.read.format("com.databricks.spark.csv") // read in keywords file
		// 	.option("delimiter", "\t")
		// 	.load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/purchasedkeywordid_tokensid.txt")
		//
		// var query_df = spark.read.format("com.databricks.spark.csv") // read in query file
		// 	.option("delimiter", "\t")
		// 	.load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/queryid_tokensid.txt")
		//
		// var title_df = spark.read.format("com.databricks.spark.csv") // read in title id
		// 	.option("delimiter", "\t")
		// 	.load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/titleid_tokensid.txt")
		//
		// var userprofile_df =  spark.read.format("com.databricks.spark.csv") // read in userprofile file
		// 	.option("delimiter", "\t")
		// 	.load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/userid_profile.txt")
		//
		// // joins to other datasets performed here, inner joins so that we get the fullest examples
		// df = train_data
		//
		// // join the query dataset
		// df = df.join(query_df, df("QueryID") <=> query_df("_c0"))
		// df = df.withColumn("QueryTokens", df("_c1")).drop("_c1","_c0")
		//
		// // join the keywords dataset
		// df = df.join(keywords_df, df("KeywordID") <=> keywords_df("_c0"))
		// df = df.withColumn("KeywordTokens", df("_c1")).drop("_c1","_c0")
		//
		// // join the title dataset
		// df = df.join(title_df, df("TitleID") <=> title_df("_c0"))
		// df = df.withColumn("TitleTokens", df("_c1")).drop("_c1","_c0")
		//
		// // join the description dataset
		// df = df.join(descriptions_df, df("DescriptionID") <=> descriptions_df("_c0"))
		// df = df.withColumn("DescriptionTokens", df("_c1")).drop("_c1","_c0")
		//
		// // join user profile dataset, with gender and age columns
		// df = df.join(userprofile_df, df("UserID") <=> userprofile_df("_c0"))
		// df = df.withColumn("Gender", df("_c1")).drop("_c1","_c0")
		// df = df.withColumn("Age", df("_c2").cast("Int")).drop("_c2")
		//
		// train_data = df
		//
        // // take rows and write to intermediate file
        // train_data.write.format("com.databricks.spark.csv").option("header", "true").save("/Users/JCFL/Code/CT/Data Science/Assignment2+3/tester")

        // read in intermediate file
        // var training_main = spark.read.format("com.databricks.spark.csv").option("header", "true") // read in main training file
        //                         .load(training_refined_location)
		//
		// var refine_ct = training_main.count()
        // // list of identifiers for creating CTR columns
        // var ctr_identifiers = Seq("DisplayURL", "AdID", "AdvertiserID", "QueryID", "KeywordID", "TitleID", "DescriptionID", "UserID")
        // var group_temp = training_main
		//
        // // simple non-combination CTR calculation for various identifiers
        // for(identifier <- ctr_identifiers) {
        //     // new column name
        //     var col_name = identifier + "_CTR"
		//
        //     // group by the identifier, and generate sum of click and impression columns
        //     group_temp = training_main.groupBy(identifier).agg(sum("Click"), sum("Impression"))
		//
        //     // calculate the ctr, join it into the main table
        //     group_temp = group_temp.withColumn(col_name, group_temp("sum(Click)")/group_temp("sum(Impression)")).drop("sum(Click)", "sum(Impression)")
        //     group_temp = group_temp.withColumn("id", group_temp(identifier)).drop(identifier)
        //     training_main = training_main.join(group_temp, training_main(identifier) <=> group_temp("id")).drop("id")
		//
        //     // normalize the CTR column
        //     training_main = training_main.withColumn(col_name + "_normd", training_main(col_name) * (training_main("Position")/(training_main("Depth")))).drop(col_name)
        // }
		//
		// group_temp = training_main.groupBy("AdID", "QueryID").agg(sum("Click"), sum("Impression"))
		// group_temp = group_temp.withColumn("MainCTR", group_temp("sum(Click)")/group_temp("sum(Impression)")).drop("sum(Click)", "sum(Impression)")
		// group_temp = group_temp.withColumn("aid", group_temp("AdID")).drop("AdID")
		// group_temp = group_temp.withColumn("qid", group_temp("QueryID")).drop("QueryID")
		//
		// training_main = training_main.join(group_temp, training_main("AdID") <=> group_temp("aid") && training_main("QueryID") <=> group_temp("qid")).drop("aid").drop("qid")
		// training_main = training_main.withColumn("MainCTRNormd", training_main("MainCTR") * (training_main("Position")/(training_main("Depth")))).drop("MainCTR")

		var training_main = spark.read.format("com.databricks.spark.csv").option("header", "true")
								.option("delimiter", "\t") // read in main training file
                                .load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/newtrain/*.csv")

        // string indexing and onehotencoding

        var string_cols = Seq("DisplayURL", "AdID", "AdvertiserID", "QueryID", "KeywordID",
                                "TitleID", "DescriptionID", "UserID", "Gender")
        var onehot_cols = Seq()
        var ctr_cols = Seq()

        // pass each string column through the indexer and one hot encoder
        for(column <- string_cols) {
            // set indexer columns
            val str_index = new StringIndexer().setInputCol(column)
                    .setOutputCol(column + "_index")
                    .fit(training_main)

            // transform
            training_main = str_index.transform(training_main)

            // set one hot encoder columns
            val onehotencoder = new OneHotEncoder().setInputCol(column + "_index")
                         .setOutputCol(column + "_onehot")

            // transform
            training_main = onehotencoder.transform(training_main)

            // add the new column to our list
            onehot_cols:+ (column + "_onehot")
            if(column != "Gender") {
                ctr_cols:+ (column + "_CTR_normd")
            }
        }
        // training_main = training_main.withColumn("AgeInt", train_data("Age").cast("Int")).drop("Age")

        // assemble all string vars
        val assembler = new VectorAssembler()
            .setInputCols(Array("UserID_onehot", "Gender_onehot", "TitleID_onehot", "QueryID_onehot", "DescriptionID_onehot", "KeywordID_onehot", "AdvertiserID_onehot", "AdID_onehot"))
            .setOutputCol("features")

        training_main = assembler.transform(training_main)

		training_main = training_main.withColumn("label", training_main("MainCTRNormd").cast("Double"))

		var simplified = training_main.select("features", "label")

		simplified.write.save("/Users/JCFL/Code/CT/Data Science/Assignment2+3/simplified.parquet")

        // training_main.write.format("com.databricks.spark.csv").option("header", "true").option("delimiter", "\t").save("/Users/JCFL/Code/CT/Data Science/Assignment2+3/newtrain")

		training_main.printSchema()
		training_main.show(20)

		val paramGrid = new ParamGridBuilder().build()

		val evaluator = new RegressionEvaluator()
		  .setLabelCol("label")
		  .setPredictionCol("prediction")
		  .setMetricName("rmse")

		val rf = new RandomForestRegressor().setLabelCol("label")
											.setFeaturesCol("features")
											.setPredictionCol("prediction")

		val cv = new CrossValidator()
		  .setEstimator(rf)
		  .setEvaluator(evaluator)
		  .setEstimatorParamMaps(paramGrid)
		  .setNumFolds(5)  // Use 3+ in practice

		val cvModel = cv.fit(training_main)

		var prediction_table = cvModel.transform(training_main)


        prediction_table.printSchema()
		prediction_table.show(20)
	}

	def refineTrainingSet(df_val: Dataset[org.apache.spark.sql.Row]): Dataset[org.apache.spark.sql.Row] = {
		val spark = SparkSession.builder()
                                .appName("CTR")
                                .getOrCreate()
		var df = df_val
		// get groupings by AdID and QueryID
        var groupedByAdAndQuery = df.groupBy("AdID", "QueryID").count().sort(desc("count"))

        var top_N_instances = groupedByAdAndQuery.limit(25000) // take top 25k instances

        // prepare for join
        top_N_instances = top_N_instances.withColumn("AID", top_N_instances("AdID")).drop("AdID")
        top_N_instances = top_N_instances.withColumn("QID", top_N_instances("QueryID")).drop("QueryID")

        // inner join based on matching AdIDs and QueryIDs, intended to filter out non
        var train_refined = df.join(top_N_instances, (df("AdID") <=> top_N_instances("AID")) && (df("QueryID") <=> top_N_instances("QID")))

        df = train_refined

		return df
	}

	def processTrainColumns(df_val: Dataset[org.apache.spark.sql.Row]): Dataset[org.apache.spark.sql.Row] = {
		val spark = SparkSession.builder()
                                .appName("CTR")
                                .getOrCreate()
		var df = df_val
		// rename all columns, cast certain columns to int
		df = df.withColumn("Click", df("_c0").cast("Int")).drop("_c0")
							.withColumn("Impression", df("_c1").cast("Int")).drop("_c1")
							.withColumn("DisplayURL", df("_c2")).drop("_c2")
							.withColumn("AdID", df("_c3")).drop("_c3")
							.withColumn("AdvertiserID", df("_c4")).drop("_c4")
							.withColumn("Depth", df("_c5").cast("Int")).drop("_c5")
							.withColumn("Position", df("_c6").cast("Int")).drop("_c6")
							.withColumn("QueryID", df("_c7")).drop("_c7")
							.withColumn("KeywordID", df("_c8")).drop("_c8")
							.withColumn("TitleID", df("_c9")).drop("_c9")
							.withColumn("DescriptionID", df("_c10")).drop("_c10")
							.withColumn("UserID", df("_c11")).drop("_c11")

		return df
	}

	def processTestColumns(df_val: Dataset[org.apache.spark.sql.Row]): Dataset[org.apache.spark.sql.Row] = {
		val spark = SparkSession.builder()
                                .appName("CTR")
                                .getOrCreate()
		var df = df_val
		// rename all columns, cast certain columns to int
		df = df.withColumn("DisplayURL", df("_c2")).drop("_c2")
							.withColumn("AdID", df("_c3")).drop("_c3")
							.withColumn("AdvertiserID", df("_c4")).drop("_c4")
							.withColumn("Depth", df("_c5").cast("Int")).drop("_c5")
							.withColumn("Position", df("_c6").cast("Int")).drop("_c6")
							.withColumn("QueryID", df("_c7")).drop("_c7")
							.withColumn("KeywordID", df("_c8")).drop("_c8")
							.withColumn("TitleID", df("_c9")).drop("_c9")
							.withColumn("DescriptionID", df("_c10")).drop("_c10")
							.withColumn("UserID", df("_c11")).drop("_c11")

		return df
	}

	def joinToOtherTables(df_val: Dataset[org.apache.spark.sql.Row]): Dataset[org.apache.spark.sql.Row] = {
		val spark = SparkSession.builder()
                                .appName("CTR")
                                .getOrCreate()
		var df = df_val
		// import additional data files
        var descriptions_df = spark.read.format("com.databricks.spark.csv") // read in descriptions file
            .option("delimiter", "\t")
            .load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/descriptionid_tokensid.txt")

        var keywords_df = spark.read.format("com.databricks.spark.csv") // read in keywords file
            .option("delimiter", "\t")
            .load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/purchasedkeywordid_tokensid.txt")

        var query_df = spark.read.format("com.databricks.spark.csv") // read in query file
            .option("delimiter", "\t")
            .load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/queryid_tokensid.txt")

        var title_df = spark.read.format("com.databricks.spark.csv") // read in title id
            .option("delimiter", "\t")
            .load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/titleid_tokensid.txt")

        var userprofile_df =  spark.read.format("com.databricks.spark.csv") // read in userprofile file
            .option("delimiter", "\t")
            .load("/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/userid_profile.txt")

		// joins to other datasets performed here, inner joins so that we get the fullest examples

        // join the query dataset
        df = df.join(query_df, df("QueryID") <=> query_df("_c0"))
        df = df.withColumn("QueryTokens", df("_c1")).drop("_c1","_c0")

        // join the keywords dataset
        df = df.join(keywords_df, df("KeywordID") <=> keywords_df("_c0"))
        df = df.withColumn("KeywordTokens", df("_c1")).drop("_c1","_c0")

        // join the title dataset
        df = df.join(title_df, df("TitleID") <=> title_df("_c0"))
        df = df.withColumn("TitleTokens", df("_c1")).drop("_c1","_c0")

        // join the description dataset
        df = df.join(descriptions_df, df("DescriptionID") <=> descriptions_df("_c0"))
        df = df.withColumn("DescriptionTokens", df("_c1")).drop("_c1","_c0")

        // join user profile dataset, with gender and age columns
        df = df.join(userprofile_df, df("UserID") <=> userprofile_df("_c0"))
        df = df.withColumn("Gender", df("_c1")).drop("_c1","_c0")
        df = df.withColumn("Age", df("_c2").cast("Int")).drop("_c2")

		return df
	}
}

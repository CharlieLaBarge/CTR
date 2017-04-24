import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}

object CTR {
	def main(args: Array[String]) {
        // store dataset locations
        val training_full_location = "/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/training.txt"
        val training_reduced_location = "/Users/JCFL/Code/CT/Data Science/Assignment2+3/data/track2/training_reduced.txt"
        val training_refined_location = "/Users/JCFL/Code/CT/Data Science/Assignment2+3/output/25k/data.csv"

        // initialize contexts
        val conf = new SparkConf().setAppName("CTR")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")

        // initiate spark session
        val spark = SparkSession.builder()
                                .appName("CTR")
                                .getOrCreate()

        // read in train data
        var train_data = spark.read.format("com.databricks.spark.csv") // read in main training file
                                        .option("delimiter", "\t")
                                        .load(training_reduced_location)

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

        // rename all columns, cast certain columns to int
        train_data = train_data.withColumn("Click", train_data("_c0").cast("Int")).drop("_c0")
                            .withColumn("Impression", train_data("_c1").cast("Int")).drop("_c1")
                            .withColumn("DisplayURL", train_data("_c2")).drop("_c2")
                            .withColumn("AdID", train_data("_c3")).drop("_c3")
                            .withColumn("AdvertiserID", train_data("_c4")).drop("_c4")
                            .withColumn("Depth", train_data("_c5").cast("Int")).drop("_c5")
                            .withColumn("Position", train_data("_c6").cast("Int")).drop("_c6")
                            .withColumn("QueryID", train_data("_c7")).drop("_c7")
                            .withColumn("KeywordID", train_data("_c8")).drop("_c8")
                            .withColumn("TitleID", train_data("_c9")).drop("_c9")
                            .withColumn("DescriptionID", train_data("_c10")).drop("_c10")
                            .withColumn("UserID", train_data("_c11")).drop("_c11")

        // get groupings by AdID and QueryID
        var groupedByAdAndQuery = train_data.groupBy("AdID", "QueryID").count().sort(desc("count"))

        var top_N_instances = groupedByAdAndQuery.limit(240) // partially arbitrary, yields around 25k

        // prepare for join
        top_N_instances = top_N_instances.withColumn("AID", top_N_instances("AdID")).drop("AdID")
        top_N_instances = top_N_instances.withColumn("QID", top_N_instances("QueryID")).drop("QueryID")

        // inner join based on matching AdIDs and QueryIDs, intended to filter out non
        var train_refined = train_data.join(top_N_instances, (train_data("AdID") <=> top_N_instances("AID")) && (train_data("QueryID") <=> top_N_instances("QID")))

        train_data = train_refined

        // joins to other datasets performed here, inner joins so that we get the fullest examples

        // join the query dataset
        train_data = train_data.join(query_df, train_data("QueryID") <=> query_df("_c0"))
        train_data = train_data.withColumn("QueryTokens", train_data("_c1")).drop("_c1","_c0")

        // join the keywords dataset
        train_data = train_data.join(keywords_df, train_data("KeywordID") <=> keywords_df("_c0"))
        train_data = train_data.withColumn("KeywordTokens", train_data("_c1")).drop("_c1","_c0")

        // join the title dataset
        train_data = train_data.join(title_df, train_data("TitleID") <=> title_df("_c0"))
        train_data = train_data.withColumn("TitleTokens", train_data("_c1")).drop("_c1","_c0")

        // join the description dataset
        train_data = train_data.join(descriptions_df, train_data("DescriptionID") <=> descriptions_df("_c0"))
        train_data = train_data.withColumn("DescriptionTokens", train_data("_c1")).drop("_c1","_c0")

        // join user profile dataset, with gender and age columns
        train_data = train_data.join(userprofile_df, train_data("UserID") <=> userprofile_df("_c0"))
        train_data = train_data.withColumn("Gender", train_data("_c1")).drop("_c1","_c0")
        train_data = train_data.withColumn("Age", train_data("_c2").cast("Int")).drop("_c2")

        // take the first 25000 rows and write to intermediate file
        train_data.limit(25000).coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("/Users/JCFL/Code/CT/Data Science/Assignment2+3/tester")

        // read in intermediate file
        var training_main = spark.read.format("com.databricks.spark.csv").option("header", "true" ) // read in main training file
                                .load(training_refined_location)

        // list of identifiers for creating CTR columns
        var ctr_identifiers = Seq("DisplayURL", "AdID", "AdvertiserID", "QueryID", "KeywordID", "TitleID", "DescriptionID", "UserID")

        var group_temp = training_main

        // simple non-combination CTR calculation for various identifiers
        for(identifier <- ctr_identifiers) {
            // new column name
            var col_name = identifier + "_CTR"

            // group by the identifier, and generate sum of click and impression columns
            group_temp = training_main.groupBy(identifier).agg(sum("Click"), sum("Impression"))

            // calculate the ctr, join it into the main table
            group_temp = group_temp.withColumn(col_name, group_temp("sum(Click)")/group_temp("sum(Impression)")).drop("sum(Click)", "sum(Impression)")
            group_temp = group_temp.withColumn("id", group_temp(identifier)).drop(identifier)
            training_main = training_main.join(group_temp, training_main(identifier) <=> group_temp("id")).drop("id")

            // normalize the CTR column
            training_main = training_main.withColumn(col_name + "_normd", training_main(col_name) * (training_main("Position")/(training_main("Depth"))))
        }

        // calculate CTRs for certain combinations of identifiers
        // for(identifier1 <- ctr_identifiers) {
        //     for(identifier2 <- ctr_identifiers) {
        //         if(identifier1 != identifier2) {
        //             var col_name = identifier1 + "&" + identifier2 + "_CTR" // column combo name
        //
        //             println("Generating CTR column for combination of " + identifier1 + " and " + identifier2)
        //
        //             group_temp = training_main.groupBy(identifier1, identifier2).agg(sum("Click"), sum("Impression"))
        //             group_temp = group_temp.withColumn(col_name, group_temp("sum(Click)")/group_temp("sum(Impression)")).drop("sum(Click)", "sum(Impression)")
        //
        //             group_temp = group_temp.withColumn("id1", group_temp(identifier1)).drop(identifier1)
        //             group_temp = group_temp.withColumn("id2", group_temp(identifier2)).drop(identifier2)
        //
        //             training_main = training_main.join(group_temp, training_main(identifier1) <=> group_temp("id1") && (training_main(identifier2) <=> group_temp("id2"))).drop("id1", "id2")
        //         }
        //     }
        // }

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

        training_main.printSchema()

        // assemble all string vars
        val assembler = new VectorAssembler()
            .setInputCols(onehot_cols.toArray)
            .setOutputCol("features")

        training_main = assembler.transform(training_main)

        training_main.limit(25000).coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("delimiter", "\t").save("/Users/JCFL/Code/CT/Data Science/Assignment2+3/newtrain")

        training_main.printSchema()
	}
}

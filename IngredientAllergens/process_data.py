from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, trim, when, expr, array, array_union, array_contains
from functools import reduce

spark = SparkSession.builder \
	.appName("App") \
	.master("local[*]") \
	.getOrCreate()

df = spark.read \
	.option("header", "true") \
	.option("inferSchema", "true") \
	.option("delimiter", "\t") \
	.csv("en.openfoodfacts.org.products.csv")

df.write.mode("overwrite").parquet("OpenFoodFacts.parquet")

df = spark.read.parquet("OpenFoodFacts.parquet")

data = df.select("code", "ingredients_text", "ingredients_tags", "ingredients_analysis_tags", "allergens", "allergens_en", "traces", "traces_tags", "traces_en", "additives", "additives_tags", "additives_en", "completeness").where((col("ingredients_text").isNotNull()) | (col("ingredients_tags").isNotNull()) | (col("ingredients_analysis_tags").isNotNull()))

cols = ["ingredients_text", "ingredients_tags", "ingredients_analysis_tags", "allergens", "allergens_en", "traces", "traces_tags", "traces_en", "additives", "additives_tags", "additives_en"]

for c in cols:
	data = data.withColumn(
		c,
		when(col(c).isNotNull(),
			expr(f"""filter(
				transform(
					split({c}, ','),
					x -> split(lower(trim(x)), ':')[size(split(trim(x), ':')) - 1]
				),
				x -> x != ''
			)""")) \
		.otherwise(array())
	)

del cols

merger_dict = {"merged_ingredients": ["ingredients_text", "ingredients_tags", "ingredients_analysis_tags", "additives_tags", "additives_en"], "merged_allergens": ["allergens", "allergens_en"], "merged_traces": ["traces", "traces_tags", "traces_en"]}

for k, v in merger_dict.items():
	data = data.withColumn(k, reduce(lambda a, b: array_union(a, b), [col(c) for c in v]))
	data = data.drop(*v)

for k in merger_dict.keys():
	classes = data.selectExpr(f"explode({k}) as tag").groupBy("tag").count().filter(col("count") >= 1000).orderBy(col("count").desc()).rdd.flatMap(lambda row: [row["tag"]]).collect()
	print(f"{k}: {len(classes)} distinct classes")
	for c in classes:
		data = data.withColumn(f"{k}_{c}", array_contains(data[k], c).cast("int"))

data = data.drop(*merger_dict.keys())

del merger_dict

data = data.drop("completeness")

data.show()

df.write.mode("overwrite").parquet("OpenFoodFacts.parquet")

spark.stop()
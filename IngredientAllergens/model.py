from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, concat, col, array_distinct, coalesce, array
from pprint import pp
from pyspark.ml.feature import CountVectorizer, VectorAssembler, StringIndexer, Binarizer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

with SparkSession.builder.appName('App').master('local[*]').config('spark.driver.memory', '4g').config('spark.executor.memory', '4g').config('spark.sql.shuffle.partitions', '64').config('spark.network.timeout', '600s').config('spark.executor.heartbeatInterval', '60s').config('spark.driver.maxResultSize', '2g').getOrCreate() as spark:
	data = spark.read.parquet('data/food.parquet')\
		.select(
			'additives_tags',
			'allergens_tags',
			'categories_tags',
			'code',
			'completeness',
			'food_groups_tags',
			'ingredients_analysis_tags',
			'ingredients_original_tags',
			'ingredients_tags',
			'labels_tags',
			'manufacturing_places_tags',
			'minerals_tags',
			'packagings.material',
			'traces_tags',
			'vitamins_tags'
		).withColumn(
			'features',
			coalesce(concat(*[col(c) for c in [
				'additives_tags',
				'ingredients_original_tags',
				'ingredients_tags',
				'manufacturing_places_tags',
				'material'
			]]), array())
		).withColumn(
			'labels',
			coalesce(concat(*[col(c) for c in [
				'allergens_tags',
				'categories_tags',
				'food_groups_tags',
				'ingredients_analysis_tags',
				'labels_tags',
				'minerals_tags',
				'traces_tags',
				'vitamins_tags'
			]]), array())
		).drop(
			'additives_tags',
			'ingredients_original_tags',
			'ingredients_tags',
			'manufacturing_places_tags',
			'material',
			'allergens_tags',
			'categories_tags',
			'food_groups_tags',
			'ingredients_analysis_tags',
			'labels_tags',
			'minerals_tags',
			'traces_tags',
			'vitamins_tags'
		)
	feature_vectorizer = CountVectorizer(
		inputCol = 'features',
		outputCol = 'featureVec'
	)
	feature_binarizer = Binarizer(
		threshold = 0.0,
		inputCol = 'featureVec',
		outputCol = 'featureBin'
	)
	label_vectorizer = CountVectorizer(
		inputCol = 'labels',
		outputCol = 'labelVec'
	)
	label_binarizer = Binarizer(
		threshold = 0.0,
		inputCol = 'labelVec',
		outputCol = 'labelBin'
	)
	pipeline = Pipeline(stages = [feature_vectorizer, feature_binarizer, label_vectorizer, label_binarizer])
	prep = pipeline.fit(data)
	data = prep.transform(data)
	label_count = len(data.select('labelVec').first()[0])
	models = []
	train, test = data.randomSplit(weights = [0.8, 0.2], seed = 1123456789)
	for label in range(label_count):
		models.append(
			LogisticRegression(
				featuresCol = 'featureVec',
				labelCol = 'label'
			).fit(
				train.withColumn(
					'label',
					vector_to_array('labelVec')[label].cast('double')
				)
			)
		)
		models[label].transform(
			test.withColumn(
				'label',
				vector_to_array('labelVec')[label].cast('double')
			)
		).show()

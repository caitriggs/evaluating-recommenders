import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import pandas as pd

# Setup a SparkSession
spark = SparkSession.builder.getOrCreate()

class Rec(object):

    def __init__(self, model):
        # Model that's already been fit
        self.model = model
        self.predictions = None

    def get_ranked_ids(self, ids, user, length):
        '''
        E.g. get_ranked_ids([1,4,7,10,15], 25, ndcg_k + len( train ))
        INPUT: a list of repo_ids to make predictions on and the user to use
        OUTPUT: a list of sorted repo_ids by recommendation prediction (descending)
        '''
        df = pd.DataFrame({'repo_id': ids, 'user_id': [user for i in range(len(ids))]})
        spark_df = spark.createDataFrame(df)
        self.predictions = self.model.transform(spark_df)
        df = self.predictions.toPandas()\
            .sort_values('prediction', ascending=False)\
            .head(length)
        return list(df['repo_id'])

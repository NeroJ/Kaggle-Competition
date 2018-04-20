import sys
import os
import time
import json
import datetime
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import *
from operator import add
from sklearn.cross_validation import KFold
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS, SVMWithSGD
from pyspark.mllib.classification import SVMModel, LogisticRegressionModel
#os.environ["PYSPARK_PYTHON"]="/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7"
#os.environ["PYSPARK_PYTHON"] = "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5"
#-----classification model parameters to be determined-------------#
class Use_Model:
    def __init__(self, Clustering=True, baseDir = 'test.csv', modelType = 'LRlbfgs', filename = 'test.csv'):
        self.filename = filename
        self.downloadName = filename+'_'+modelType +'_predicted.csv'
        self.baseDir = baseDir
        self.sc = SparkContext()
        self.spark = SparkSession \
            .builder \
            .appName("Fraud_Detection") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        self.df = self.spark.read.csv(self.baseDir+'/uploads/'+self.filename, header=True, inferSchema=True).drop('click_id')
        self.header = self.df.columns
        self.rdd = self.sc.parallelize(self.df.collect())
        #-----must be a local variable------#
        def time_Parse(rdd_row):
            weekday_ = rdd_row['click_time'].timetuple().tm_wday
            hour_ = rdd_row['click_time'].timetuple().tm_hour
            new_rddRow = list(rdd_row) + [weekday_, hour_]
            return tuple(new_rddRow)
        self.new_data = self.rdd.map(lambda x: time_Parse(x)).collect()
        #-----function end-----------------#
        self.new_header = self.header + ['weekday', 'hour']
        self.rdd2 = self.sc.parallelize(self.new_data)
        self.df_new = self.spark.createDataFrame(self.rdd2, self.new_header)
        self.df_PD = self.df_new.select('ip','app','device','os','channel','weekday','hour').distinct().toPandas()
        #--------KMeans Clustering----------#
        if Clustering:
            self.KMeans_Processing(['weekday','hour'])
        #-----Features&Prediction-----#
        self.Features = np.array(self.df_PD[list(self.df_PD.columns)])
        self.Prediction(modelType)
        #-----to.csv-----#
        self.df_PD.to_csv(self.baseDir + '/uploads/'+self.downloadName)

    def KMeans_Processing(self, columns):
        data_point = np.array(self.df_PD[columns])
        model = KMeansModel.load(self.sc, self.baseDir+'/fraudModel/Model/'+'KMeans')
        result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
        self.df_PD.insert(len(list(self.df_PD.columns)), 'KMeans_feature', result)

    def Prediction(self, modelType):
        data_point = self.Features
        if modelType == 'RF':
            model = RandomForestModel.load(self.sc, self.baseDir+'/fraudModel/Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        elif modelType == 'GBDT':
            model = GradientBoostedTreesModel.load(self.sc, self.baseDir+'/fraudModel/Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        elif modelType == 'LRsgd':
            model = LogisticRegressionModel.load(self.sc, self.baseDir+'/fraudModel/Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        elif modelType == 'LRlbfgs':
            model = LogisticRegressionModel.load(self.sc, self.baseDir+'/fraudModel/Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        elif modelType == 'SVM':
            model = SVMModel.load(self.sc, self.baseDir+'/fraudModel/Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        else:
            pass

    def __del__(self):
        self.sc.stop()
        self.spark.stop()

import sys
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import *
from operator import add
from sklearn.cross_validation import KFold
from pyspark.mllib.clustering import KMeans, KMeansModel
import os
import time
import datetime
from math import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
os.environ["PYSPARK_PYTHON"]="/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7"

class Fraud_Detection:
    def __init__(self, Clustering=False, file_name='train_sample.csv'):
        self.sc = SparkContext()
        self.spark = SparkSession \
            .builder \
            .appName("Fraud_Detection") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        self.df = self.spark.read.csv(file_name, header=True, inferSchema=True).drop('attributed_time')
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
        self.df_PD = self.df_new.select('ip','app','device','os','channel','weekday','hour','is_attributed').distinct().toPandas()
        #--------KMeans Clustering----------#
        if KMeans:
            self.KMeans_Processing(['weekday','hour'], 10)
        #-----Training-----#
        self.Features = np.array(self.df_PD[list(self.df_PD.columns)[:-1]])
        self.Lables = np.array(self.df_PD[list(self.df_PD.columns)[-1]])
        self.Training_Validation()

    def KMeans_Processing(self, columns, k):
        data_point = np.array(self.df_PD[columns])
        model = KMeans.train(self.sc.parallelize(data_point), k, maxIterations=100, initializationMode="random",
                            seed=50, initializationSteps=2, epsilon=1e-4)
        result = np.array([model.predict(item) for item in data_point])
        self.df_PD.insert(len(list(self.df_PD.columns))-1, 'KMeans_feature', result)

    def Validation_Accuracy(self, XInput_Train, YInput_Train, XInput_Vali, YInput_Vali):
        dataInput = [LabeledPoint(item1, item2) for item1, item2 in zip(YInput_Train, XInput_Train)]
        model = RandomForest.trainClassifier(data=self.sc.parallelize(dataInput),
                                             numClasses=2,categoricalFeaturesInfo={},
                                             numTrees=10,maxDepth=30,seed=42)
        Y_pre = [model.predict(item) for item in XInput_Vali]
        counter = 0
        for pre, val in zip(Y_pre, YInput_Vali):
            if pre == float(val):
                counter += 1
        return float(counter)/float(len(Y_pre))

    def Training_Validation(self):
        self.kf = KFold(n=len(self.Features), n_folds=5, shuffle=True)
        counter = 0
        Acc_accumulator = 0
        for tr, tst in self.kf:
            acc = self.Validation_Accuracy(self.Features[tr], self.Lables[tr], self.Features[tst], self.Lables[tst])
            counter += 1
            Acc_accumulator += acc
            print 'The '+str(counter)+'st validation accuracy: ' + str(acc)
            break
        print 'The average validation accuracy: ' +str(Acc_accumulator/float(counter))

    def __del__(self):
        self.sc.stop()
        self.spark.stop()


Fraud_Detection(Clustering=True, file_name='train_sample.csv')

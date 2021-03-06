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
Parameters = {'RF':{'categoricalFeaturesInfo':{}, 'numTrees':20,'maxDepth':30},
              'GBDT': {'categoricalFeaturesInfo':{}, 'learningRate':0.1, 'numIterations':100, 'loss':'leastSquaresError'},
              'LRsgd':{'iterations':100, 'step':0.1, 'miniBatchFraction':1.0, 'regParam':0.1, 'regType':'l2'},
              'LRlbfgs':{'iterations':100, 'regParam':0.0, 'regType':'l2'},
              'SVM':{'iterations':100, 'step':0.1, 'regParam':0.01, 'miniBatchFraction':1.0, 'regType':'l2'},
              'KMeans':{'k':5, 'maxIterations':100, 'initializationMode':'random', 'seed':50, 'initializationSteps':2, 'epsilon':1e-4}}

class Fraud_DetectionTraining:
    def __init__(self, Clustering=True, file_name='train_sample.csv', classificationModel='LRlbfgs', ModelSave=True, classificationPara=Parameters):
        self.modelnameDir = [item for item in os.listdir('Model')]
        self.Parameters = Parameters
        self.file = open('Result_txt/'+classificationModel+'_Result', 'w')
        self.sc = SparkContext()
        self.spark = SparkSession \
            .builder \
            .appName("Fraud_Detection") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        self.df = self.spark.read.csv(file_name[0], header=True, inferSchema=True).drop('attributed_time')
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
        if Clustering:
            self.KMeans_Processing(['weekday','hour'])
        #-----Training-----#
        self.Features = np.array(self.df_PD[list(self.df_PD.columns)[:-1]])
        self.Lables = np.array(self.df_PD[list(self.df_PD.columns)[-1]])
        self.Training_Validation(classificationModel)
        #---model saving---#
        if ModelSave == True:
            self.Model_Saving(classificationModel)

    def split_process(self, name):
        dfList = [self.spark.read.csv(name[i], header=True, inferSchema=True).drop('attributed_time') for i in range(len(NameList))]
        temp = []
        for item in dfList:
            temp = temp + item.collect()
        df_ = self.spark.createDataFrame(temp)
        return df_

    def KMeans_Processing(self, columns):
        data_point = np.array(self.df_PD[columns])
        if 'KMeans' not in self.modelnameDir:
            model = KMeans.train(self.sc.parallelize(data_point), k=self.Parameters['KMeans']['k'],
                                 maxIterations=self.Parameters['KMeans']['maxIterations'],
                                 initializationMode=self.Parameters['KMeans']['initializationMode'],
                                 seed=self.Parameters['KMeans']['seed'],
                                 initializationSteps=self.Parameters['KMeans']['initializationSteps'],
                                 epsilon=self.Parameters['KMeans']['epsilon'])
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns))-1, 'KMeans_feature', result)
            model.save(self.sc, 'Model/'+'KMeans')
        else:
            model = KMeansModel.load(self.sc, 'Model/'+'KMeans')
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns))-1, 'KMeans_feature', result)
        paras = json.dumps(Parameters['KMeans'])
        self.file.write('KMeans Parameters: '+str(paras)+'\n')

    def Validation_Accuracy(self, XInput_Train, YInput_Train, XInput_Vali, YInput_Vali, modelType):
        dataInput = [LabeledPoint(item1, item2) for item1, item2 in zip(YInput_Train, XInput_Train)]
        if modelType == 'RF':
            model = RandomForest.trainClassifier(data=self.sc.parallelize(dataInput),
                                                 numClasses=2,categoricalFeaturesInfo=self.Parameters[modelType]['categoricalFeaturesInfo'],
                                                 numTrees=self.Parameters[modelType]['numTrees'],
                                                 maxDepth=self.Parameters[modelType]['maxDepth'],
                                                 seed=42)
            paras = json.dumps(Parameters[modelType])
            self.file.write(modelType+' Parameters: '+str(paras)+'\n')
        elif modelType == 'GBDT':
            model = GradientBoostedTrees.trainClassifier(data=self.sc.parallelize(dataInput),
                                                        learningRate=self.Parameters[modelType]['learningRate'],
                                                        categoricalFeaturesInfo=self.Parameters[modelType]['categoricalFeaturesInfo'],
                                                        numIterations=self.Parameters[modelType]['numIterations'],
                                                        loss=self.Parameters[modelType]['loss'])
            paras = json.dumps(Parameters[modelType])
            self.file.write(modelType+' Parameters: '+str(paras)+'\n')
        elif modelType == 'LRsgd':
            model = LogisticRegressionWithSGD.train(self.sc.parallelize(dataInput),
                                                    iterations=self.Parameters[modelType]['iterations'],
                                                    step=self.Parameters[modelType]['step'],
                                                    miniBatchFraction=self.Parameters[modelType]['miniBatchFraction'],
                                                    regParam=self.Parameters[modelType]['regParam'],
                                                    regType=self.Parameters[modelType]['regType'])
            paras = json.dumps(Parameters[modelType])
            self.file.write(modelType+' Parameters: '+str(paras)+'\n')
        elif modelType == 'LRlbfgs':
            model = LogisticRegressionWithLBFGS.train(self.sc.parallelize(dataInput),
                                                      iterations=self.Parameters[modelType]['iterations'],
                                                      regParam=self.Parameters[modelType]['regParam'],
                                                      regType=self.Parameters[modelType]['regType'])
            paras = json.dumps(Parameters[modelType])
            self.file.write(modelType+' Parameters: '+str(paras)+'\n')
        elif modelType == 'SVM':
            model = SVMWithSGD.train(self.sc.parallelize(dataInput),
                                     iterations=self.Parameters[modelType]['iterations'],
                                     step=self.Parameters[modelType]['step'],
                                     regParam=self.Parameters[modelType]['regParam'],
                                     miniBatchFraction=self.Parameters[modelType]['miniBatchFraction'],
                                     regType=self.Parameters[modelType]['regType'])
            paras = json.dumps(Parameters[modelType])
            self.file.write(modelType+' Parameters: '+str(paras)+'\n')
        else:
            pass
        Y_pre = model.predict(self.sc.parallelize(XInput_Vali)).collect()
        def count_acc(X):
            if X[0] == float(X[1]):
                return ('Yes', 1)
            else:
                return ('No', 1)
        Yes_No = self.sc.parallelize(zip(Y_pre, YInput_Vali)).map(lambda x: count_acc(x)).reduceByKey(add).collect()
        for item in Yes_No:
            if item[0] == 'Yes':
                return float(item[1])/float(len(Y_pre))
            else:
                pass

    def Training_Validation(self, model):
        self.kf = KFold(n=len(self.Features), n_folds=5, shuffle=True)
        counter = 0
        Acc_accumulator = 0
        for tr, tst in self.kf:
            acc = self.Validation_Accuracy(self.Features[tr], self.Lables[tr], self.Features[tst], self.Lables[tst], model)
            counter += 1
            Acc_accumulator += acc
            print ('The '+str(counter)+'st validation accuracy of '+model+' : '+str(acc))
            self.file.write('The '+str(counter)+'st validation accuracy of '+model+' : '+str(acc)+'\n')
            break
        print ('The average validation accuracy of '+model+' : ' +str(Acc_accumulator/float(counter)))
        self.file.write('The average validation accuracy of '+model+' : ' +str(Acc_accumulator/float(counter))+'\n')

    def Model_Saving(self, modelType):
        dataInput = [LabeledPoint(item1, item2) for item1, item2 in zip(self.Lables, self.Features)]
        if modelType == 'RF':
            if modelType not in self.modelnameDir:
                model = RandomForest.trainClassifier(data=self.sc.parallelize(dataInput),
                                                     numClasses=2,
                                                     categoricalFeaturesInfo=self.Parameters[modelType]['categoricalFeaturesInfo'],
                                                     numTrees=self.Parameters[modelType]['numTrees'],
                                                     maxDepth=self.Parameters[modelType]['maxDepth'],
                                                     seed=42)
                model.save(self.sc, 'Model/'+modelType)
        elif modelType == 'GBDT':
            if modelType not in self.modelnameDir:
                model = GradientBoostedTrees.trainClassifier(data=self.sc.parallelize(dataInput),
                                                            learningRate=self.Parameters[modelType]['learningRate'],
                                                            categoricalFeaturesInfo=self.Parameters[modelType]['categoricalFeaturesInfo'],
                                                            numIterations=self.Parameters[modelType]['numIterations'],
                                                            loss=self.Parameters[modelType]['loss'])
                model.save(self.sc, 'Model/'+modelType)
        elif modelType == 'LRsgd':
            if modelType not in self.modelnameDir:
                model = LogisticRegressionWithSGD.train(self.sc.parallelize(dataInput),
                                                        iterations=self.Parameters[modelType]['iterations'],
                                                        step=self.Parameters[modelType]['step'],
                                                        miniBatchFraction=self.Parameters[modelType]['miniBatchFraction'],
                                                        regParam=self.Parameters[modelType]['regParam'],
                                                        regType=self.Parameters[modelType]['regType'])
                model.save(self.sc, 'Model/'+modelType)
        elif modelType == 'LRlbfgs':
            if modelType not in self.modelnameDir:
                model = LogisticRegressionWithLBFGS.train(self.sc.parallelize(dataInput),
                                                          iterations=self.Parameters[modelType]['iterations'],
                                                          regParam=self.Parameters[modelType]['regParam'],
                                                          regType=self.Parameters[modelType]['regType'])
                model.save(self.sc, 'Model/'+modelType)
        elif modelType == 'SVM':
            if modelType not in self.modelnameDir:
                model = SVMWithSGD.train(self.sc.parallelize(dataInput),
                                         iterations=self.Parameters[modelType]['iterations'],
                                         step=self.Parameters[modelType]['step'],
                                         regParam=self.Parameters[modelType]['regParam'],
                                         miniBatchFraction=self.Parameters[modelType]['miniBatchFraction'],
                                         regType=self.Parameters[modelType]['regType'])
                model.save(self.sc, 'Model/'+modelType)
        else:
            pass

    def __del__(self):
        self.file.close()
        self.sc.stop()
        self.spark.stop()

class Use_Model:
    def __init__(self, Clustering=True, file_name = 'test.csv', modelType = 'LRlbfgs'):
        self.sc = SparkContext()
        self.spark = SparkSession \
            .builder \
            .appName("Fraud_Detection") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        self.df = self.spark.read.csv(file_name, header=True, inferSchema=True).drop('click_id')
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
        self.df_PD.to_csv('predicted.csv')

    def KMeans_Processing(self, columns):
        data_point = np.array(self.df_PD[columns])
        model = KMeansModel.load(self.sc, 'Model/'+'KMeans')
        result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
        self.df_PD.insert(len(list(self.df_PD.columns)), 'KMeans_feature', result)

    def Prediction(self, modelType):
        data_point = self.Features
        if modelType == 'RF':
            model = RandomForestModel.load(self.sc, 'Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        elif modelType == 'GBDT':
            model = GradientBoostedTreesModel.load(self.sc, 'Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        elif modelType == 'LRsgd':
            model = LogisticRegressionModel.load(self.sc, 'Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        elif modelType == 'LRlbfgs':
            model = LogisticRegressionModel.load(self.sc, 'Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        elif modelType == 'SVM':
            model = SVMModel.load(self.sc, 'Model/'+modelType)
            result = np.array(model.predict(self.sc.parallelize(data_point)).collect())
            self.df_PD.insert(len(list(self.df_PD.columns)), 'result', result)
        else:
            pass

    def __del__(self):
        self.sc.stop()
        self.spark.stop()

Use_Model(Clustering=True, file_name = 'test.csv', modelType = 'LRlbfgs')

def fitModel():
    NameList = ['xaaaa.csv']
    test_list = ['LRlbfgs']
    for item in test_list:
        print ('Runing Model: '+ item)
        Fraud_DetectionTraining(Clustering=True, file_name = NameList, classificationModel=item, ModelSave=True)

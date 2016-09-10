'''
Created on Nov 11, 2014

@author: chenxh
'''
#encoding=utf-8
import os,numpy,theano,cPickle,sys,time, sklearn
import theano.tensor as T
from preprocess_data import *
from dualDBN import *
from evaluation import *
from reconstructedDBN import *
from svmutil import *
from som_enhanced import *
from minisom import MiniSom
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as LR, Perceptron
from sklearn import tree,svm

def trainTdbn(finetune_lr=0.1,pretraining_epochs=200,
              pretrain_lr=0.01,k=1,training_epochs=200,batch_size=10,dataIndex=0):
    
    start_time = time.clock()
    filepath = "/home/chenxh/workspace/SpammerDetection/data/featureVector/JointFeatureResult.txt"
    saveFile = file("TianChi_result2.txt",'a')
    (train_set_x,train_set_y,train_x,train_y),(test_set_x,test_set_y,test_x,test_y) = Rdata_load(filepath,
                                    dataIndex*1500,(dataIndex+1)*1500)
#                                     2,6)
#                                     0,0)
#     (topic_train, topic_label,topic_x,topic_y) = Ldata_load("/home/chenxh/workspace/DBNprocessing/dataset/features/dlda_1207.fword")
    print train_set_x.get_value(borrow=True).shape[1]
    print test_set_x.get_value(borrow=True).shape[1]
     
    n_ins = train_set_x.get_value(borrow=True).shape[1]
    
    r_datasets = [(train_set_x,train_set_y),(test_set_x,test_set_y)]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_out = 2
   
    print '..... building the right and left model'
    rDBN = DBN(n_ins=n_ins,hidden_layers_sizes=[n_ins],n_outs=n_out)
            
            
    pretrainingR_fns, OutputLayer = rDBN.pretraining_function(train_set_x, batch_size, k)
            
    for i in xrange(rDBN.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    end_time = time.clock()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_prob, r_LayerOutput = rDBN.build_finetune_functions(
                datasets=r_datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
    predict_y, origin_y = r_get_test_label()  
    print 'predict_y: ',predict_y
    print 'probability: ',shape(r_LayerOutput())
    print 'Round ',dataIndex,',DBN precision,recall,F1: ',evaluation(predict_y, origin_y)
    print >>saveFile, 'Round ',dataIndex,',DBN precision,recall,F1: ',evaluation(predict_y, origin_y)
#     print 'output Layer:', shape(OutputLayer())
#     print evaluation(predict_y, origin_y)
   
    
#     filepath1=unicode(filepath,'utf8')
#     y, x1 = svm_read_dataset(filepath1)
    
    
#     x = r_LayerOutput()
#     x1 = train_x
#     y = train_y
#     fold = 10
#     ave = len(x)/fold
#     avePre = []
#     aveRecall= []
#     aveF1 = []
#     for i in xrange(fold):
#         x_train = numpy.concatenate((x[:i*ave],x[(i+1)*ave:]))
#         y_train = y[:i*ave]+y[(i+1)*ave:]
#         x_test = x[i*ave:(i+1)*ave]
#         y_test = y[i*ave:(i+1)*ave]
#         clf = svm.SVC(kernel = 'linear', C=1000)
#         clf.fit(x_train, y_train)
#         precision,recall,F1 = evaluation(clf.predict(x_test),y_test)
#         print 'Round ',i,',DBN precision,recall,F1: ',precision,recall,F1
#         avePre.append(precision)
#         aveRecall.append(recall)
#         aveF1.append(F1)

#         x_train = x1[:i*ave]+x1[(i+1)*ave:]
#         x_test = x1[i*ave:(i+1)*ave]
#         print 'beginning...',shape(x_train)
#         clf.fit(x_train, y_train)
#         print 'ending...'
#         precision,recall,F1 = evaluation(clf.predict(x_test),y_test)
#         print 'Round ',i,',SVM precision,recall,F1: ',precision,recall,F1
#     print  numpy.mean(avePre),numpy.mean(aveRecall),numpy.mean(aveF1)
    
#     trainX = r_LayerOutput()[1499:]
#     trainY = train_y[1499:]
#     testX = r_LayerOutput()[:1499]
#     testY = train_y[:1499]
    
    
#     m = svm_train(trainY, trainX, '-c 1000 -b 1')
#     print 'beginning...'
#     p_label, p_acc, p_val = svm_predict(testY,testX, m,"-b 1")
#     print shape(p_val),p_label[0],p_val[0]
#     print evaluation(p_label, testY)
    
    end_time = time.clock()
    print 'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print 'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print '------------------------------------------------------------------------------'
def Lauch(dataIndex=0):
    filepath = "/home/chenxh/workspace/SpammerDetection/data/featureVector/JointFeatureResult.txt"
    saveFile = file("Spammer_result.txt",'a')
    (train_set_x1,train_set_y1,train_x,train_y),(test_set_x1,test_set_y1,test_x,test_y) = Rdata_load(filepath,
#                                     dataIndex*1500,(dataIndex+1)*1500,
                                     0,0,
                                     0,2)
    (train_set_x2,train_set_y2,train_x,train_y),(test_set_x2,test_set_y2,test_x,test_y) = Rdata_load(filepath,
#                                     dataIndex*1500,(dataIndex+1)*1500,
                                     0,0,
                                     2,6)
    (train_set_x3,train_set_y3,train_x,train_y),(test_set_x3,test_set_y3,test_x,test_y) = Rdata_load(filepath,
#                                     dataIndex*1500,(dataIndex+1)*1500,
                                     0,0,
                                     6,None)
    
    ContentOutput = train1DBN(train_set_x = train_set_x3,train_set_y=train_set_y3,test_set_x=test_set_x3,test_set_y=test_set_y3)
    timeOutput = train1DBN(train_set_x = train_set_x1,train_set_y=train_set_y1,test_set_x=test_set_x1,test_set_y=test_set_y1)
    AttributeOutput = train1DBN(train_set_x = train_set_x2,train_set_y=train_set_y2,test_set_x=test_set_x2,test_set_y=test_set_y2)
     
    CombinedFeature = []
    if len(timeOutput)== len(AttributeOutput) and len(timeOutput)== len(ContentOutput):
        for i in xrange(len(timeOutput)):
            CombinedFeature.append(append((append(timeOutput[i],AttributeOutput[i])),ContentOutput[i]))
    print shape(CombinedFeature)
#     shared_x = theano.shared(numpy.asarray(CombinedFeature,dtype=theano.config.floatX),borrow=True)
#     precision,recall,F1 = trainCDBN(train_set_x = shared_x,train_set_y=train_set_y3,test_set_x=test_set_x3,test_set_y=test_set_y3)
    
    x = CombinedFeature
    y = train_y
    fold = 10
    ave = len(x)/fold
    avePre = []
    aveRecall= []
    aveF1 = []
    for i in xrange(fold):
#         x_train = numpy.concatenate((x[:i*ave],x[(i+1)*ave:]))
        x_train = x[:i*ave]+x[(i+1)*ave:]
        y_train = y[:i*ave]+y[(i+1)*ave:]
        x_test = x[i*ave:(i+1)*ave]
        y_test = y[i*ave:(i+1)*ave]
        
        shared_Train_x = theano.shared(numpy.asarray(x_train,dtype=theano.config.floatX),borrow=True)
        shared_Train_y = theano.shared(numpy.asarray(y_train,dtype=theano.config.floatX),borrow=True)
        shared_Train_y = T.cast(shared_Train_y,'int32')
        shared_test_x = theano.shared(numpy.asarray(x_test,dtype=theano.config.floatX),borrow=True)
        shared_test_y = theano.shared(numpy.asarray(y_test,dtype=theano.config.floatX),borrow=True)
        shared_test_y = T.cast(shared_test_y,'int32')
        
        precision,recall,F1 = trainCDBN(train_set_x = shared_Train_x,train_set_y=shared_Train_y,test_set_x=shared_test_x,test_set_y=shared_test_y)
    
        
#         clf = svm.SVC(kernel = 'linear', C=1000)
#         clf.fit(x_train, y_train)
#         precision,recall,F1 = evaluation(clf.predict(x_test),y_test)
        print >>saveFile,'Round ',i,',DBN precision,recall,F1: ',precision,recall,F1
        avePre.append(precision)
        aveRecall.append(recall)
        aveF1.append(F1)
  
    print  >>saveFile, numpy.mean(avePre),numpy.mean(aveRecall),numpy.mean(aveF1)
    
    
def train1DBN(finetune_lr=0.1,pretraining_epochs=500,
              pretrain_lr=0.01,k=1,training_epochs=200,batch_size=10,train_set_x=None,train_set_y=None,test_set_x=None,test_set_y=None):
    
    start_time = time.clock()
    print train_set_x.get_value(borrow=True).shape[0]
    print train_set_x.get_value(borrow=True).shape[1]
    n_out = 2
      
    r_datasets = [(train_set_x,train_set_y),(train_set_x,train_set_y)]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_ins=train_set_x.get_value(borrow=True).shape[1]
   
    print '..... building the right and left model'
    rDBN = DBN(n_ins=n_ins,hidden_layers_sizes=[n_ins],n_outs=n_out)
            
            
    pretrainingR_fns, OutputLayer = rDBN.pretraining_function(train_set_x, batch_size, k)
            
    for i in xrange(rDBN.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    end_time = time.clock()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_prob, r_LayerOutput = rDBN.build_finetune_functions(
                datasets=r_datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
#     predict_y, origin_y = r_get_test_label()  
#     for i in predict_y:
#         print >> saveFile, i
#     for i in r_features():
#         print >> saveFile, i
#     print 'predict_y: ',predict_y
#     print 'probability: ',shape(r_features())
    print 'output Layer:', shape(OutputLayer())
#     print evaluation(predict_y, origin_y)
   
    
    end_time = time.clock()
    print 'Finish one type DBN  %.2f mins' % ((end_time - start_time) / 60.)
    print '------------------------------------------------------------------------------'
    return r_LayerOutput()

def trainCDBN(finetune_lr=0.1,pretraining_epochs=500,
              pretrain_lr=0.01,k=1,training_epochs=200,batch_size=10,train_set_x=None,train_set_y=None,test_set_x=None,test_set_y=None):
    
    start_time = time.clock()
    print train_set_x.get_value(borrow=True).shape[0]
    print train_set_x.get_value(borrow=True).shape[1]
    n_out = 2
      
    r_datasets = [(train_set_x,train_set_y),(test_set_x,test_set_y)]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_ins=train_set_x.get_value(borrow=True).shape[1]
   
    print '..... building the right and left model'
    rDBN = DBN(n_ins=n_ins,hidden_layers_sizes=[n_ins,n_ins/2],n_outs=n_out)
            
            
    pretrainingR_fns, OutputLayer = rDBN.pretraining_function(train_set_x, batch_size, k)
            
    for i in xrange(rDBN.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    end_time = time.clock()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_prob, r_LayerOutput = rDBN.build_finetune_functions(
                datasets=r_datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
    predict_y, origin_y = r_get_test_label()  
#     for i in predict_y:
#         print >> saveFile, i
#     for i in r_features():
#         print >> saveFile, i
#     print 'predict_y: ',predict_y
#     print 'probability: ',shape(r_features())
#     print 'output Layer:', shape(OutputLayer())
    precision,recall,F1 = evaluation(predict_y, origin_y) 
    print 'Round ',i,',DBN precision,recall,F1: ',precision,recall,F1
#     avePre = []
#     aveRecall = []
#     aveF1 = []
#     avePre.append(precision)
#     aveRecall.append(recall)
#     aveF1.append(F1)
  
    
    end_time = time.clock()
    print 'Finish CDBN  %.2f mins' % ((end_time - start_time) / 60.)
    print '------------------------------------------------------------------------------'
    return precision,recall,F1
Lauch()
# trainTdbn()
# for ind in xrange(10):
#     trainTdbn(dataIndex=ind)
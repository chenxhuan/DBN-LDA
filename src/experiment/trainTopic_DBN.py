'''
Created on Nov 11, 2014

@author: chenxh
'''
#encoding=utf-8
import os,numpy,theano,cPickle,sys,time, sklearn
sys.path.append("..")
import theano.tensor as T
from preprocess.preprocess_data import *
from model.DBN import *
from evaluation import *
from model.svmutil import *
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as LR, Perceptron
from sklearn import tree

def trainTdbn(finetune_lr=0.1,pretraining_epochs=200,
              pretrain_lr=0.1,k=1,training_epochs=200,batch_size=10,dataIndex=0, lamda=0.05):
    
    start_time = time.time()
    # filepath = "../../dataset/features/annotation1000_20160927"
    filepath = "../../dataset/features/mixed_5_20161007"
    saveFile = file("../../output/result2.txt",'a')
    print >> saveFile, 'round ', dataIndex, 'lamda ', lamda
    start_index = 0
    end_index = 116
    fold_size = 116
    (train_set_x,train_set_y,train_x,train_y),(test_set_x,test_set_y,test_x,test_y) = Rdata_load(filepath,
                                    dataIndex*fold_size,(dataIndex+1)*fold_size)
                                    # start_index,end_index)
    (topic_train, topic_label,topic_x,topic_y) = Ldata_load("../../dataset/features/lexicon2_20160928")
    print train_set_x.get_value(borrow=True).shape[0]
    print test_set_x.get_value(borrow=True).shape[0]

    # build the topic set
    topic_list_set = []
    for y in train_y:
        topic_list_set.append(topic_x[y])
    topic_sets = theano.shared(numpy.asarray(topic_list_set,dtype=theano.config.floatX),borrow=True)
     
    r_datasets = [(train_set_x,train_set_y),(test_set_x,test_set_y)]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_out = 26

    print '..... building the topic supervised DBN'
    tDBN = DBN(n_ins=1188,supervised_type=1, hidden_layers_sizes=[594,594,594],n_outs=n_out)

    pretrainingR_fns,Layers = tDBN.pretraining_function(train_set_x=train_set_x,supervised_set=topic_sets, batch_size=batch_size, k=k)

    for i in xrange(tDBN.n_layers):
        for epoch in xrange(pretraining_epochs if i is not 1 else 2*pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr, ld=lamda))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c,axis=0)
    end_time = time.time()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_features, Layers = tDBN.build_finetune_functions(
        datasets=r_datasets, batch_size=batch_size,
        learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        c = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
            c.append(minibatch_avg_cost1)
        print 'fine-tuning epoch %d, cost ' % epoch, numpy.mean(c)
    predict_y, origin_y = r_get_test_label()
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print 'test results from topic supervised DBN , presion, recall, F1, accuracy: '
    # print >> saveFile,'test results from topic supervised DBN, presion, recall, F1, accuracy: '
    resutl = evaluation(predict_y, origin_y)
    print >> saveFile,resutl[0],'\t',resutl[1],'\t',resutl[2],'\t',getAccuracy(predict_y, origin_y)
    print 'test results from topic supervised DBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)

    print '..... building the label supervised DBN'
    dbn1 = DBN(n_ins=1188,supervised_type=2, hidden_layers_sizes=[594,594,594],n_outs=n_out)
    pretrainingR_fns,Layers = dbn1.pretraining_function(train_set_x=train_set_x, supervised_set=train_set_y, batch_size=batch_size, k=k)
    for i in xrange(dbn1.n_layers):
        for epoch in xrange(pretraining_epochs if i is not 1 else 2*pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr, ld=1))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c,axis=0)
    end_time = time.time()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_features, Layers = dbn1.build_finetune_functions(
        datasets=r_datasets, batch_size=batch_size,
        learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        c = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
            c.append(minibatch_avg_cost1)
        print 'fine-tuning epoch %d, cost ' % epoch, numpy.mean(c)

    predict_y, origin_y = r_get_test_label()
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print 'test results from label supervised DBN , presion, recall, F1, accuracy: '
    # print >> saveFile,'test results from dual supervised DBN, presion, recall, F1, accuracy: '
    resutl = evaluation(predict_y, origin_y)
    print >> saveFile,resutl[0],'\t',resutl[1],'\t',resutl[2],'\t',getAccuracy(predict_y, origin_y)
    print 'test results from label supervised DBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)




    print '..... building the DBN model'
    rDBN = DBN(n_ins=1188,hidden_layers_sizes=[594,594,594],n_outs=n_out)

    pretrainingR_fns,Layers = rDBN.pretraining_function(train_set_x=train_set_x, batch_size=batch_size, k=k)

    for i in xrange(rDBN.n_layers):
        for epoch in xrange(pretraining_epochs if i is not 1 else 2*pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c,axis=0)
    end_time = time.time()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_features, Layers = rDBN.build_finetune_functions(
                datasets=r_datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        c = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
            c.append(minibatch_avg_cost1)
        print 'fine-tuning epoch %d, cost ' % epoch,numpy.mean(c)
    predict_y, origin_y = r_get_test_label()
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print 'second results from  DBN , presion, recall, F1, accuracy: '
    # print >> saveFile,'first results from right DBN , presion, recall, F1, accuracy: '
    resutl = evaluation(predict_y, origin_y)
    print >> saveFile,resutl[0],'\t',resutl[1],'\t',resutl[2],'\t',getAccuracy(predict_y, origin_y)
    print 'second results from  DBN , presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)


    print 'third results from GNaiveBayes , presion, recall, F1, accuracy: '
    # print  >> saveFile,'third results from GNaiveBayes , presion, recall, F1, accuracy: '
    nb = GaussianNB()
    X = numpy.array(train_x)
    Y = numpy.array(train_y)
    nb.fit(X, Y)
    nbResult = nb.predict(test_x)
    # print >> saveFile,'predict_y: ',nbResult
    nbResult = change2PrimaryC(nbResult)
    test_y = change2PrimaryC(test_y)
    resutl = evaluation(nbResult, test_y)
    print >> saveFile,resutl[0],'\t',resutl[1],'\t',resutl[2],'\t',getAccuracy(nbResult, test_y)

    print 'fourth results from SVM, presion,recall,F1,accuracy'
    # print >> saveFile,'fourth results from SVM, presion,recall,F1,accuracy'


    filepath1=unicode(filepath,'utf8')
    y, x = svm_read_dataset(filepath1)
    trainX = x[:dataIndex*fold_size]+x[(dataIndex+1)*fold_size:]
    trainY = y[:dataIndex*fold_size]+y[(dataIndex+1)*fold_size:]
    testX = x[dataIndex*fold_size:(dataIndex+1)*fold_size]
    testY = y[dataIndex*fold_size:(dataIndex+1)*fold_size]
    # trainX = x[:start_index]+x[end_index:]
    # trainY = y[:start_index]+y[end_index:]
    # testX = x[start_index:end_index]
    # testY = y[start_index:end_index]

    m = svm_train(trainY, trainX, '-c 10')
    p_label, p_acc, p_val = svm_predict(testY,testX, m)
    p_label = change2PrimaryC(p_label)
    # print >> saveFile,'predict_y: ',p_label
    testY = change2PrimaryC(testY)
    resutl = evaluation(p_label, testY)
    print >> saveFile,resutl[0],'\t',resutl[1],'\t',resutl[2],'\t',getAccuracy(p_label, testY)

    print 'fifth results from Perception, presion,recall,F1,accuracy'
    # print >> saveFile,'fifth results from Perception, presion,recall,F1,accuracy'
    per = Perceptron()
    per.fit(X, Y)
    perceptronResult = per.predict(test_x)
    perceptronResult = change2PrimaryC(perceptronResult)
    resutl = evaluation(perceptronResult, test_y)
    print >> saveFile,resutl[0],'\t',resutl[1],'\t',resutl[2],'\t',getAccuracy(perceptronResult, test_y)

    print 'sixth results from Decision Trees , presion, recall, F1, accuracy: '
    # print >> saveFile,'sixth results from Decision Trees, presion,recall,F1,accuracy'
    dTree = tree.DecisionTreeClassifier()
    dTree.fit(X, Y)
    TreeResult = dTree.predict(test_x)
    # print >> saveFile,'predict_y: ',TreeResult
    TreeResult = change2PrimaryC(TreeResult)
    resutl = evaluation(TreeResult, test_y)
    print >> saveFile,resutl[0],'\t',resutl[1],'\t',resutl[2],'\t',getAccuracy(TreeResult, test_y)
    end_time = time.time()
    print >> saveFile,'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print 'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print >>saveFile,'------------------------------------------------------------------------------'

# lds = [1,0.5,0.1,0.05,0.01,0.001]
# for ld in lds:
#     print 'ld ', ld
#     trainTdbn(dataIndex=7,lamda=ld)
# trainTdbn(dataIndex=1)
for ind in xrange(10):
    print 'round ', ind
    trainTdbn(dataIndex=ind)

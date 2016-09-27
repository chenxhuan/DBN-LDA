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
              pretrain_lr=0.01,k=1,training_epochs=200,batch_size=10,dataIndex=0):
    
    start_time = time.clock()
    filepath = "../../dataset/features/tnbz_annotation"
    # filepath = "../../dataset/features/mixed_2015-03-25"
    saveFile = file("../../output/result2.txt",'a')
    (train_set_x,train_set_y,train_x,train_y),(test_set_x,test_set_y,test_x,test_y) = Rdata_load(filepath,
                                    # dataIndex*1569,(dataIndex+1)*1569)
                                    0,1000)
    (topic_train, topic_label,topic_x,topic_y) = Ldata_load("../../dataset/features/dlda_1207.fword")
    print train_set_x.get_value(borrow=True).shape[0]
    print test_set_x.get_value(borrow=True).shape[0]
    
     
    r_datasets = [(train_set_x,train_set_y),(test_set_x,test_set_y)]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_out = 26
  
    print '..... building the right and left model'
    rDBN = DBN(n_ins=1188,hidden_layers_sizes=[594,594,594],n_outs=n_out)
           
    pretrainingR_fns,Layers = rDBN.pretraining_function(train_set_x, batch_size, k)
           
    for i in xrange(rDBN.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    end_time = time.clock()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_features, Layers = rDBN.build_finetune_functions(
                datasets=r_datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
    predict_y, origin_y = r_get_test_label()  
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print 'first results from right DBN , presion, recall, F1, accuracy: '
    print >> saveFile,'first results from right DBN , presion, recall, F1, accuracy: '
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
          
    r_sigmoid_layers,r_output,r_params = rDBN.getParams(train_set_x)
    r_sigmoid_layersT,r_outputT,r_paramsT = rDBN.getParams(test_set_x)
    print shape(r_sigmoid_layers())
           
    new_features = []
    l_features = []
    new_featuresT = []
    for x in topic_x:
        l_features = append(l_features,x)
#     l_features = [l_sigmoid_layers()[i] for i in xrange(len_left)]
# new_features = map(list,zip(*(map(list,zip(*a))+map(list,zip(*b)))))
    print shape(l_features)
    for r_feature in r_sigmoid_layers():
        new_features.append(append(l_features,r_feature))
    for r_feature in r_sigmoid_layersT():
        new_featuresT.append(append(l_features,r_feature))
    print shape(new_features),shape(new_featuresT)
           
#     sim_result =  similarity(r_sigmoid_layers(),topic_x)
#     new_features = []
#     for i in xrange(len(sim_result)):
#         feature = append(topic_x[sim_result[i]],(r_sigmoid_layers()[i]))
#         new_features.append(feature)
#     sim_result =  similarity(r_sigmoid_layersT(),topic_x)
#     new_featuresT = []
#     for i in xrange(len(sim_result)):
#         feature = append(topic_x[sim_result[i]],(r_sigmoid_layersT()[i]))
#         new_featuresT.append(feature)
#     print shape(new_features),shape(new_featuresT)
           
            
    n_dim = shape(new_features)[1]
    shared_x = theano.shared(numpy.asarray(new_features,dtype=theano.config.floatX),borrow=True)
    shared_y = theano.shared(numpy.asarray(new_featuresT,dtype=theano.config.floatX),borrow=True)
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=[n_dim/4],n_outs=26)
    pretraining_fns, Layers = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features, Layers = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
#     print >> saveFile,'origin_y: ',origin_y
#     print >> saveFile,'predict_y: ',predict_y
    print >> saveFile,evaluation(predict_y, origin_y), getAccuracy(predict_y, origin_y)
#     sigmoid_layers,output,params = dbn.getParams(shared_x)
#     print shape(sigmoid_layers())
    # reconstructed right data 
 
    print 'third results from GNaiveBayes , presion, recall, F1, accuracy: '
    print  >> saveFile,'third results from GNaiveBayes , presion, recall, F1, accuracy: '
    nb = GaussianNB()
    X = numpy.array(train_x)
    Y = numpy.array(train_y)
    nb.fit(X, Y)
    nbResult = nb.predict(test_x)
    # print >> saveFile,'predict_y: ',nbResult
    nbResult = change2PrimaryC(nbResult)
    test_y = change2PrimaryC(test_y)
    test_precision,test_recall,F1 = evaluation(nbResult, test_y)
    print >> saveFile,(test_precision,test_recall,F1, getAccuracy(nbResult, test_y))
    print 'fourth results from LogisticRegression , presion, recall, F1, accuracy: '
    print  >> saveFile,'third results from LogisticRegression , presion, recall, F1, accuracy: '
    lr = LR()
    lr.fit(X, Y)
    lrResult = lr.predict(test_x)
    # print >> saveFile,'predict_y: ',lrResult
    lrResult = change2PrimaryC(lrResult)
    test_precision,test_recall,F1 = evaluation(lrResult, test_y)
    print >> saveFile,(test_precision,test_recall,F1, getAccuracy(lrResult, test_y))
     
      
    print 'fifth results from SVM, presion,recall,F1,accuracy'
    print >> saveFile,'fifth results from SVM, presion,recall,F1,accuracy'
      
      
    filepath1=unicode(filepath,'utf8')
    y, x = svm_read_dataset(filepath1)
    # trainX = x[:dataIndex*1569]+x[(dataIndex+1)*1569:]
    # trainY = y[:dataIndex*1569]+y[(dataIndex+1)*1569:]
    # testX = x[dataIndex*1569:(dataIndex+1)*1569]
    # testY = y[dataIndex*1569:(dataIndex+1)*1569]
    trainX =x[100:]
    trainY=y[100:]
    testX=x[:100]
    testY=y[:100]
     
    m = svm_train(trainY, trainX, '-c 1')
    p_label, p_acc, p_val = svm_predict(testY,testX, m)
    p_label = change2PrimaryC(p_label)
    # print >> saveFile,'predict_y: ',p_label
    testY = change2PrimaryC(testY)
    test_precision,test_recall,F1 = evaluation(p_label, testY)
    print >>saveFile,(test_precision,test_recall,F1,getAccuracy(p_label, testY))
    print 'sixth results from Perception, presion,recall,F1,accuracy'
    print >> saveFile,'sixth results from Perception, presion,recall,F1,accuracy'
    per = Perceptron()
    per.fit(X, Y)
    perceptronResult = per.predict(test_x)
    perceptronResult = change2PrimaryC(perceptronResult)
    test_precision,test_recall,F1 = evaluation(perceptronResult, test_y)
    print >> saveFile,(test_precision,test_recall,F1, getAccuracy(perceptronResult, test_y))
    print 'seventh results from Decision Trees , presion, recall, F1, accuracy: '
    print >> saveFile,'seventh results from Decision Trees, presion,recall,F1,accuracy'
    dTree = tree.DecisionTreeClassifier()
    dTree.fit(X, Y)
    TreeResult = dTree.predict(test_x)
    # print >> saveFile,'predict_y: ',TreeResult
    TreeResult = change2PrimaryC(TreeResult)
    test_precision,test_recall,F1 = evaluation(TreeResult, test_y)
    print >> saveFile,(test_precision,test_recall,F1, getAccuracy(TreeResult, test_y))
    end_time = time.clock()
    print >> saveFile,'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print 'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print >>saveFile,'------------------------------------------------------------------------------'
    
trainTdbn()
# for ind in range(2):
#     trainTdbn(dataIndex=ind+8)
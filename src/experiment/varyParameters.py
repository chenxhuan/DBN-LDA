'''
Created on Nov 27, 2014

@author: chenxh
'''
#encoding=utf-8
import os,numpy,theano,cPickle,sys,time, sklearn
import theano.tensor as T
from src.preprocess.preprocess_data import *
from src.model.dualDBN import *
from evaluation import *
from svmutil import *
from som_enhanced import *
from minisom import MiniSom
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as LR, Perceptron
from sklearn import tree

saveFile = file("../../output/result_parameter.txt",'a')
def trainTdbn(finetune_lr=0.1,pretraining_epochs=200,
              pretrain_lr=0.01,k=1,training_epochs=200,batch_size=10,dataIndex=9,hidden_layers=[594,594,594]):
   
    start_time = time.clock()
    filepath = "../../dataset/features/anno1000_feature_2014-11-17"
    
    (train_set_x,train_set_y,train_x,train_y),(test_set_x,test_set_y,test_x,test_y) = Rdata_load(filepath,
                                    dataIndex*100,(dataIndex+1)*100)
#                                     10,3936)
    (topic_train, topic_label,topic_x,topic_y) = Ldata_load("../../dataset/features/dlda_1207.fword")
    print train_set_x.get_value(borrow=True).shape[0]
    print test_set_x.get_value(borrow=True).shape[0]
    
     
    r_datasets = [(train_set_x,train_set_y),(test_set_x,test_set_y)]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_out = 26
  
    print '..... building the right and left model'
    rDBN = DBN(n_ins=1188,hidden_layers_sizes=hidden_layers,n_outs=n_out)
           
           
    pretrainingR_fns = rDBN.pretraining_function(train_set_x, batch_size, k)
           
    for i in xrange(rDBN.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    end_time = time.clock()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_features = rDBN.build_finetune_functions(
                datasets=r_datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
    predict_y, origin_y = r_get_test_label()  
    print 'first results from right DBN , presion, recall, F1, accuracy: '
    print >> saveFile,'origin_y: ',origin_y
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print >> saveFile,'first results from right DBN , presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
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
    
    layers=[100]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[200]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
 
    layers=[300]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[400]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[500]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[600]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[700]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[800]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[900]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[1000]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    
    layers=[1100]
    print  >> saveFile,'top layer hidden units: ',layers
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=layers,n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_y,test_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    predict_y, origin_y = get_test_label()  
    print 'second results from LDADBN, presion, recall, F1, accuracy: '
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from LDADBN, presion, recall, F1, accuracy: '
    print evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
 
    
    end_time = time.clock()
    print >> saveFile,'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print 'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print >>saveFile,'------------------------------------------------------------------------------'
# layers = []
# i = 100
# while i<1200:
#        
#     j = i
#     i += 100    
#     for k in xrange(3):
#         layers.append(j)
#            
#     print layers
#     print  >> saveFile,'layers: ',layers
#     trainTdbn(hidden_layers=layers)
#     layers = []
trainTdbn()
        

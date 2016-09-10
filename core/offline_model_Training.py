'''
Created on Mar 17, 2015

@author: chenxh
'''
#encoding=utf-8
import os,numpy,theano,cPickle,sys,time
import theano.tensor as T
from preprocess_data import *
from dualDBN import *
from evaluation import *

def trainLDADBN(finetune_lr = 0.1, pretraining_epochs = 200, pretrain_lr = 0.01, k = 1,
                training_epochs = 200, batch_size = 10, dataIndex = 0):
    start_time = time.clock()
    training_file_path = "/home/chenxh/workspace/DBNprocessing/dataset/features/anno1000_feature_2014-11-17"
    RightParameters_file_path = "rightParameters1"
    TopParameters_file_path = "TopParameters1"
    
    (train_set_x,train_set_y,train_x,train_y),(test_set_x,test_set_y,test_x,test_y) = Rdata_load(training_file_path,0,0)
    (topic_train, topic_label,topic_x,topic_y) = Ldata_load("/home/chenxh/workspace/DBNprocessing/dataset/features/dlda_1207.fword")
    print train_set_x.get_value(borrow=True).shape[0]
    print test_set_x.get_value(borrow=True).shape[0]
    
    r_datasets = [(train_set_x,train_set_y),(train_set_x,train_set_y)]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_out = 26
  
    print '..... building the right model'
    rDBN = DBN(n_ins=1188,hidden_layers_sizes=[594,594,594],n_outs=n_out)
           
           
    pretrainingR_fns = rDBN.pretraining_function(train_set_x, batch_size, k)
           
    for i in xrange(rDBN.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    r_train_fn, r_test_score, r_get_test_label, r_features = rDBN.build_finetune_functions(
                datasets=r_datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
    rDBN.save_params(RightParameters_file_path)
    
    print '..... building the top model'
    r_sigmoid_layers,r_output,r_params = rDBN.getParams(train_set_x)
           
    new_features = []
    l_features = []
    for x in topic_x:
        l_features = append(l_features,x)
    print shape(l_features)
    for r_feature in r_sigmoid_layers():
        new_features.append(append(l_features,r_feature))
    print shape(new_features)
    
    n_dim = shape(new_features)[1]
    shared_x = theano.shared(numpy.asarray(new_features,dtype=theano.config.floatX),borrow=True)
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=[n_dim/2,n_dim/2,n_dim/2],n_outs=26)
    pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
#             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#             print numpy.mean(c)
    datasets = [(shared_x,train_set_y),(shared_x,train_set_y)]
    train_fn, test_score, get_test_label, features = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
    dbn.save_params(TopParameters_file_path)
    end_time = time.clock()
    print "finish the training process and save the parameters... ... ...for %.2f mins" % ((end_time - start_time)/ 60.)

trainLDADBN()
    
        
    

    
    
    
    
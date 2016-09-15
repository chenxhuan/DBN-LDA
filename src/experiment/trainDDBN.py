'''
Created on Sep 30, 2014

@author: chenxh
'''
#encoding=utf-8
import os,numpy,theano,cPickle,sys,time
import theano.tensor as T
from src.preprocess.preprocess_data import *
from src.model.dualDBN import *
from evaluation import *
from src.model.reconstructedDBN import *
from svmutil import *
from src.model.som_enhanced import *
from minisom import MiniSom

def trainDdbn(finetune_lr=0.1,pretraining_epochs=200,
              pretrain_lr=0.01,k=1,training_epochs=200,batch_size=10,dataIndex=0):
    def getAccuracy(v1,v2):
        total_correct = 0
        for v, y in zip(v1, v2):
            if y == v: 
                total_correct += 1
        return 1.0*total_correct/len(v1)
    
    saveFile = file("../../output/result.txt",'a')
    (train_set_x,train_set_y,train_x,train_y),(test_set_x,test_set_y,test_x,test_y) = Rdata_load("../../dataset/features/anno1000_feature_2014-11-17",
                                    dataIndex*100,(dataIndex+1)*100)
#                                     100,3936)
    (topic_train, topic_label,topic_x,topic_y) = Ldata_load("../../dataset/features/topicFeature_2014-11-21")
    print train_set_x.get_value(borrow=True).shape[0]
    print test_set_x.get_value(borrow=True).shape[0]
    
     
    r_datasets = [(train_set_x,train_set_y),(test_set_x,test_set_y)]
    l_datasets = [(topic_train,topic_label),(topic_train,topic_label)]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_out = 26
  
    print '..... building the right and left model'
    rDBN = DBN(n_ins=1188,hidden_layers_sizes=[594,594,594],n_outs=n_out)
    lDBN = DBN(n_ins=1188,hidden_layers_sizes=[596/n_out],n_outs=n_out)
          
          
    pretrainingR_fns = rDBN.pretraining_function(train_set_x, batch_size, k)
    pretrainingL_fns = lDBN.pretraining_function(topic_train, 13, k)
          
    start_time = time.clock()
    for i in xrange(rDBN.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrainingR_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    for i in xrange(lDBN.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(2):
                c.append(pretrainingL_fns[i](index=batch_index,lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    end_time = time.clock()
    print 'Training ran for %.2f mins' % ((end_time - start_time) / 60.)
    r_train_fn, r_test_score, r_get_test_label, r_features = rDBN.build_finetune_functions(
                datasets=r_datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    l_train_fn, l_test_score, l_get_test_label, l_features = lDBN.build_finetune_functions(
                datasets=l_datasets, batch_size=13,
                learning_rate=finetune_lr)
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost1 = r_train_fn(minibatch_index)
        for minibatch_index in xrange(2):
            minibatch_avg_cost2 = l_train_fn(minibatch_index)
    predict_y, origin_y = r_get_test_label()  
#     print predict_y, origin_y
    print 'first results from right DBN , presion, recall, F1, accuracy: '
    print >> saveFile,'origin_y: ',origin_y
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print >> saveFile,'first results from right DBN , presion, recall, F1, accuracy: '
    print >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
          
         
         
    r_sigmoid_layers,r_output,r_params = rDBN.getParams(train_set_x)
    r_sigmoid_layersT,r_outputT,r_paramsT = rDBN.getParams(test_set_x)
    l_sigmoid_layers,l_output,l_params = lDBN.getParams(topic_train) 
    print shape(r_sigmoid_layers())
          
    new_features = []
    l_features = []
    new_featuresT = []
    len_left = len(l_sigmoid_layers())
    for i in xrange(len_left):
        l_features = append(l_features,l_sigmoid_layers()[i])
#     l_features = [l_sigmoid_layers()[i] for i in xrange(len_left)]
    print shape(l_features)
    for r_feature in r_sigmoid_layers():
        new_features.append(append(l_features,r_feature))
    for r_feature in r_sigmoid_layersT():
        new_featuresT.append(append(l_features,r_feature))
    print shape(new_features),shape(new_featuresT)
          
#     sim_result =  similarity(r_sigmoid_layers(),l_sigmoid_layers())
#     new_features = []
#     for i in xrange(len(sim_result)):
#         feature = append(l_sigmoid_layers()[sim_result[i]],(r_sigmoid_layers()[i]))
#         new_features.append(feature)
#     sim_result =  similarity(r_sigmoid_layersT(),l_sigmoid_layers())
#     new_featuresT = []
#     for i in xrange(len(sim_result)):
#         feature = append(l_sigmoid_layers()[sim_result[i]],(r_sigmoid_layersT()[i]))
#         new_featuresT.append(feature)
#     print shape(new_features),shape(new_featuresT)
          
           
    n_dim = shape(new_features)[1]
    shared_x = theano.shared(numpy.asarray(new_features,dtype=theano.config.floatX),borrow=True)
    shared_y = theano.shared(numpy.asarray(new_featuresT,dtype=theano.config.floatX),borrow=True)
    dbn = DBN(n_ins=n_dim, hidden_layers_sizes=[n_dim],n_outs=26)
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
    print 'second results from DDBN, presion, recall, F1, accuracy: '
    print >> saveFile,'origin_y: ',origin_y
    print >> saveFile,'predict_y: ',predict_y
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    print  >> saveFile,'second results from DDBN, presion, recall, F1, accuracy: '
    print  >> saveFile,evaluation(predict_y, origin_y),getAccuracy(predict_y, origin_y)
#     sigmoid_layers,output,params = dbn.getParams(shared_x)
#     print shape(sigmoid_layers())
    # reconstructed right data 
#     print 'third results from origin similarity , presion, recall, F1, accuracy: '
#     print >> saveFile,'third results from origin similarity , presion, recall, F1, accuracy: '
# #     rec = ReC(n_ins=n_dim,hidden_layers_sizes=[594, 594, 594], n_outs=1188, top_params=params, params=r_params[0:6])
# #     shared_x = theano.shared(numpy.asarray(sigmoid_layers(),dtype=theano.config.floatX),borrow=True)
# #     recVector = rec.getReconstructedOutput(shared_x)
# #     for i in xrange(len(train_x)):
# #         print PearsonCorrelation(train_x[i],recVector()[i]),rootMeanSquareError(train_x[i],recVector()[i])
#      
#     sim_result =  similarity(test_x,topic_x)
#     print sim_result
#     test_precision,test_recall,F1 = evaluation(sim_result, test_y)
#     print  >> saveFile,(test_precision,test_recall,F1, getAccuracy(sim_result, test_y))
       
    end_time = time.clock()
    print >> saveFile,'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print 'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    print >>saveFile,'------------------------------------------------------------------------------'
       
#     print 'fourth results from SVM, presion,recall,F1,accuracy'
#     filepath = "/home/chenxh/workspace/DBNprocessing/dataset/features/Test"
#     filepath=unicode(filepath,'utf8')
#     y, x = svm_read_dataset(filepath)
#     trainX = x[:dataIndex*2000]+x[(dataIndex+1)*2000:]
#     trainY = y[:dataIndex*2000]+y[(dataIndex+1)*2000:]
#     testX = x[dataIndex*2000:(dataIndex+1)*2000]
#     testY = y[dataIndex*2000:(dataIndex+1)*2000]
# #     trainX =x[:10165]
# #     trainY=y[:10165]
# #     testX=x[10165:]
# #     testY=y[10165:]
#    
#     m = svm_train(trainY, trainX, '-c 10')
#     p_label, p_acc, p_val = svm_predict(testY,testX, m)
#     test_precision,test_recall,F1 = evaluation(p_label, testY)
#     print >>saveFile,(test_precision,test_recall,F1,p_acc[0])
# #     print >>saveFile,'------------------------------------------------------------------------------'
# 
#     print 'fifth results from SOM, presion,recall,F1,accuracy'
#     som = MiniSom(100,100,1188,sigma=0.3,learning_rate=0.05)
#     print "Training...",shape(train_x)
#     som.train_random(train_x,training_epochs*2) # trains the SOM with 100 iterations
#     wmap = {}
#     for i,x in enumerate(train_x):
#         w = som.winner(x)
#         wmap[w] = train_y[i]
#     print len(wmap)
#     somRes = []
#     for t in test_x:
#         if som.winner(t) in wmap:
#             somRes.append(wmap[som.winner(t)])
#         else:
#             somRes.append(-1)
#     print len(somRes),somRes
#     test_precision,test_recall,F1 = evaluation(somRes, test_y)
#     print >> saveFile,test_precision,test_recall,F1,getAccuracy(somRes, test_y)
#     print >>saveFile,'------------------------------------------------------------------------------'
    
#     print 'second results from similarity DDBN, presion, recall, F1, accuracy: ',r_params[0:6]
#     rec = ReC(n_ins=n_dim,hidden_layers_sizes=[594, 594, 594], n_outs=1188, top_params=params, params=r_params[0:6])
#     shared_x = theano.shared(numpy.asarray(sigmoid_layers(),dtype=theano.config.floatX),borrow=True)
#     recVector = rec.getReconstructedOutput(shared_x)
# #     for i in xrange(len(train_x)):
# #         print PearsonCorrelation(train_x[i],recVector()[i]),rootMeanSquareError(train_x[i],recVector()[i])
#    
#     sim_result =  similarity(recVector(),topic_x)
#     print sim_result
#     test_precision,test_recall,F1 = evaluation(sim_result, train_y)
#     print test_precision,test_recall,F1
    
    
    
#     print '.....getting the finetuning functions'
#     
#     r_train_fn, r_test_model, r_get_test_label, r_features = rDBN.build_finetune_functions(
#                 datasets=r_datasets, batch_size=batch_size,
#                 learning_rate=finetune_lr)
#     l_train_fn, l_test_model, l_get_test_label, l_features = lDBN.build_finetune_functions(
#                 datasets=l_datasets, batch_size=batch_size,
#                 learning_rate=finetune_lr)
#     print '... finetunning the model'
#     
#     start_time = time.clock()
#     epoch = 0
#     while (epoch < training_epochs):
#         epoch = epoch + 1
#         for minibatch_index in xrange(n_train_batches):
#             minibatch_avg_cost = r_train_fn(minibatch_index)
#         for minibatch_index in xrange(4): 
#             minibatch_avg_cost = l_train_fn(minibatch_index)
#     end_time = time.clock()
#     print 'Fine-tuning ran for %.2f mins' % ((end_time - start_time) / 60.)
#     
#     print 'first results from original DBN, presion, recall, F1, accuracy: '
#     test_losses = r_test_model()
#     test_accuracy = numpy.mean(test_losses)
#     predict_y, origin_y = r_get_test_label()
#     test_precision,test_recall,F1 = evaluation(predict_y, origin_y)
#     print test_precision,test_recall,F1,test_accuracy
#     
#     
#     
#     print 'second results from similarity DBN, presion, recall, F1, accuracy: '
#     sim_result =  similarity(train_x,topic_x)
#     print sim_result
#     test_precision,test_recall,F1 = evaluation(sim_result, origin_y)
#     print test_precision,test_recall,F1
#     
#     print 'third results from similarity DBN, presion, recall, F1, accuracy: '
#     sim_result =  distance(r_sigmoid_layers(),l_sigmoid_layers())
#     print sim_result
#     test_precision,test_recall,F1 = evaluation(sim_result, origin_y)
#     print test_precision,test_recall,F1
#     
#     new_features = []
#     for i in xrange(len(sim_result)):
#         feature = append(l_sigmoid_layers()[sim_result[i]],(r_sigmoid_layers()[i]))
# #         feature = (l_features()[sim_result[i]])*((r_sigmoid_layers()[i]))
#         new_features.append(feature)
#     print "=====",shape(new_features)
#     input =  T.matrix('x')
#     shared_x = theano.shared(numpy.asarray(new_features,dtype=theano.config.floatX),borrow=True)
#     reconstructed_layer = HiddenLayer(rng=numpy.random.RandomState(123),
#                                         input=input,
#                                         n_in=1188,
#                                         n_out=1188,
#                                         activation=T.nnet.sigmoid, W=l_params[0],b=l_params[1])
#     getReconstructedOutput = reconstructed_layer.Output()
#     index = T.lscalar('index')
#     getLayers = theano.function([index], getReconstructedOutput,
#                    givens={input: shared_x[index :]})
#     dis_result = similarity(getLayers(0),topic_x)
#     print '```````',dis_result
#     print 'fourth results from reconstructed similarity DBN, presion, recall, F1, accuracy: '
#     test_precision,test_recall,F1 = evaluation(dis_result, origin_y)
#     print test_precision,test_recall,F1    
#     
#     print 'fifth results from  distance , presion, recall, F1, accuracy: '
# 
#     dis_result = distance(r_sigmoid_layers(),l_sigmoid_layers())
#     print (dis_result)
#     test_precision,test_recall,F1 = evaluation(dis_result, origin_y)
#     print test_precision,test_recall,F1,test_accuracy     
    
#     new_features = []
#     new_train_y = []
#     r_output = r_sigmoid_layers()
#     l_output = l_sigmoid_layers()
#     rlen = len(r_output)
#     llen = len(l_output)
#     for i in xrange(rlen):
#         for j in xrange(llen):
#             feature = append(l_output[j],(r_output[i]))
#             new_train_y.append(train_y[i])
#             new_features.append(feature)
#     print "=====",shape(new_features),shape(new_train_y)
#     shared_x = theano.shared(numpy.asarray(new_features,dtype=theano.config.floatX),borrow=True)
#     shared_y = theano.shared(numpy.asarray(new_train_y,dtype=theano.config.floatX),borrow=True)
#     dbn = DBN(n_ins=1188,hidden_layers_sizes=[594],n_outs=40)
#     pretraining_fns = dbn.pretraining_function(shared_x, batch_size, k)
#     for i in xrange(dbn.n_layers):
#         for epoch in xrange(pretraining_epochs*2):
#             c = []
#             for batch_index in xrange(n_train_batches):
#                 c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
# #             print 'Final Pre-training layer %i, epoch %d, cost ' % (i, epoch),
# #             print numpy.mean(c)
#     datasets = [(shared_x,T.cast(shared_y,'int32')),(shared_x,T.cast(shared_y,'int32'))]
#     train_fn, test_model, get_test_label, features = dbn.build_finetune_functions(
#                 datasets=datasets, batch_size=batch_size,
#                 learning_rate=finetune_lr)
#     while (epoch < training_epochs):
#         epoch = epoch + 1
#         for minibatch_index in xrange(n_train_batches):
#             minibatch_avg_cost = train_fn(minibatch_index)
# #     predict_y, origin_y = get_test_label()  
#     print 'sixth results from joint distance DBN, presion, recall, F1, accuracy:'
#     sigmoid_layers,output,params = dbn.getParams(shared_x)
#     dis_result = distance1(sigmoid_layers())
#     print (dis_result)
#     test_precision,test_recall,F1 = evaluation(dis_result, origin_y)
#     print test_precision,test_recall,F1,test_accuracy   
    
#     sim_result =  similarity(r_sigmoid_layers(),l_features())
#     print sim_result
#     new_features = []
#     for i in xrange(len(sim_result)):
#         feature = append(l_features()[sim_result[i]],(r_sigmoid_layers()[i]))
# #         feature = (l_features()[sim_result[i]])*((r_sigmoid_layers()[i]))
#         new_features.append(feature)
#     print "=====",shape(new_features)
    
    
#     x = T.matrix('x')
#     LRoutput = LogisticRegression(
#             input=x,
#             n_in=1000,
#             n_out=40, W = r_params[len(r_params)-2], b = r_params[-1])
#     predict = LRoutput.getPredict(x)
#     getPredict = theano.function([x],predict)
      

# trainDdbn()
for ind in range(10):
    trainDdbn(dataIndex=ind)




# def data_load(dataset, fi = 0, ti = 0):
#     print '... loading right data'
#     f = open(dataset,'rb')
#     lines = f.readlines()
#     lenLine = len(lines)
#     print lenLine
#     data_x = []
#     x = []
#     for line in lines:
#         line = line.strip()
#         if line=="":
#             continue
#         dataArray = []
#         for val in line.split(' '):
#             dataArray.append(float(val))
#         data_x.append(array(dataArray))
#     print shape(data_x)
#     
#     return data_x
#         
# r = data_load("/home/chenxh/workspace/DBNprocessing/dataset/features/TFIDFfeature_2014-10-03")
# l = data_load("/home/chenxh/workspace/DBNprocessing/dataset/features/topicsFeature_2014-10-03")
# results = similarity(r,l)
# print results
# save_file = open('outputResults_dir', 'w') 
# for item in results:
#     save_file.write(str(item)+"\n")
# save_file.close()
    
    
    
    
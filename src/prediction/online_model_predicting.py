'''
Created on Mar 20, 2015

@author: chenxh
'''
from src.preprocess.preprocess_data import *
from testing_with_savePares import *
from src.experiment.evaluation import *

def getTopicName(index):
    
    f = open('../../dataset/topics','rb')
    lines = f.readlines()
    return lines[index]

def predictTopic():
    
    test_file_path = "../../dataset/features/anno1000_feature_2014-11-17"
    RightParameters_file_path = "../../output/rightParameters"
    TopParameters_file_path = "../../output/TopParameters"
    
    (train_set_x,train_set_y,train_x,train_y),(test_set_x,test_set_y,test_x,test_y) = Rdata_load(test_file_path,2,3)
    (topic_train, topic_label,topic_x,topic_y) = Ldata_load("../../dataset/features/dlda_1207.fword")
    
    rDBNtest = Testing(n_ins=1188, hidden_layers_sizes=[594,594,594],n_outs=26,path = RightParameters_file_path)
    test_label,layer_output, outputVectors = rDBNtest.predictLabel(test_set_x,test_set_y)
    
    
    new_features = []
    print shape(topic_x)
    for r_feature in layer_output():
        new_features.append(append(topic_x,r_feature))
    print shape(new_features)
    
    n_dim = shape(new_features)[1]
    new_shared_x = theano.shared(numpy.asarray(new_features,dtype=theano.config.floatX),borrow=True)
    tDBNtest = Testing(n_ins=n_dim, hidden_layers_sizes=[n_dim/2,n_dim/2,n_dim/2],n_outs=26,path = TopParameters_file_path)
    test_label,layer_output, outputVectors = tDBNtest.predictLabel(new_shared_x,test_set_y)
    
    predict_y, origin_y = test_label()
    print outputVectors(),predict_y
    print getTopicName(predict_y)
    # test_precision,test_recall,F1 = evaluation(predict_y, origin_y)
    # print test_precision,test_recall,F1
    
    
    
predictTopic()
    

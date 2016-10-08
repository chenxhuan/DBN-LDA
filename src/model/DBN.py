'''
Created on Sep 29, 2014

@author: chenxh
'''
#encoding=utf-8
import os,numpy,theano,cPickle,sys
sys.path.append("..")
import theano.tensor as T
from theano import *
from theano.tensor.shared_randomstreams import RandomStreams
from rbm_supervised import *
from preprocess.preprocess_data import *

class DBN(object):
    def __init__(self,theano_rng=None,n_ins=1000, supervised_type=None,
                 hidden_layers_sizes=[594,594,594],n_outs=40):
        self.n_ins = n_ins
        self.n_outs = n_outs
        self.supervised_type = supervised_type
        self.sigmoid_layers = []
        self.sigmoid_topic_layers = []
        self.rbm_layers = []
        self.params = []
        self.hidden_layers_sizes =hidden_layers_sizes
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0
        numpy_rng = numpy.random.RandomState(123)
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        self.topic = T.matrix('topic')

        for i in xrange(self.n_layers):
            if i==0:
                input_size = n_ins
                layer_input = self.x
                topic_input = self.topic
            else:
                input_size = hidden_layers_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output
                topic_input = self.sigmoid_topic_layers[-1].output
            # get W and b
            sigmoid_layer = HiddenLayer(rng=numpy_rng,input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            sigmoid_topic_layer = HiddenLayer(rng=numpy_rng,input=topic_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid,
                                        W=sigmoid_layer.W,
                                        b=sigmoid_layer.b)
            self.sigmoid_layers.append(sigmoid_layer)
            self.sigmoid_topic_layers.append(sigmoid_topic_layer)
            self.params.extend(sigmoid_layer.params)
            if supervised_type==None:
                rbm_layer = RBM(numpy_rng=numpy_rng,theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            if supervised_type==1:
                rbm_layer = RBM(numpy_rng=numpy_rng,theano_rng=theano_rng, topic_input=topic_input,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            if supervised_type==2:
                rbm_layer = RBM(numpy_rng=numpy_rng,theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b,
                                n_out=self.n_outs,
                                y=self.y)
            self.rbm_layers.append(rbm_layer)
        self.getLayerOutput = self.sigmoid_layers[-1].Output()
        
        self.logLayer = LogisticRegression(self.sigmoid_layers[-1].output,
                                           n_in=self.hidden_layers_sizes[-1],n_out=self.n_outs)
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)
        self.label = self.logLayer.getLabel(self.y)
        self.feature = self.logLayer.getFeature()
        
    
    def pretraining_function(self,train_set_x=None,supervised_set=None, batch_size=None,k=1):
        
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')
        lamda = T.scalar('ld')

        n_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
        batch_begin = index*batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            monitoring_cost, cost,updates = rbm.get_cost_updates(learning_rate,persistent=None,k=k,type=self.supervised_type, lamda=lamda)
            if self.supervised_type==None:
                print 'training DBN'
                fn = theano.function(inputs=[index,learning_rate],
                                 outputs = [monitoring_cost, cost],
                                 updates = updates,
                                 givens = {self.x:train_set_x[batch_begin:batch_end]})
            if self.supervised_type == 1:
                print 'training domain supervised DBN'
                fn = theano.function(inputs=[index,learning_rate,lamda],
                                     outputs = [monitoring_cost, cost],
                                     updates = updates,
                                     givens = {self.x:train_set_x[batch_begin:batch_end], self.topic:supervised_set[batch_begin:batch_end]})
                                     # givens = {self.x:train_set_x[batch_begin:batch_end]})
            if self.supervised_type == 2:
                print 'training label supervised DBN'
                fn = theano.function(inputs=[index,learning_rate,lamda],
                                     outputs = [monitoring_cost, cost],
                                     updates = updates,
                                     givens = {self.x:train_set_x[batch_begin:batch_end], self.y: supervised_set[batch_begin:batch_end]})
                                     # givens = {self.x:train_set_x[batch_begin:batch_end]})
            pretrain_fns.append(fn)
        getLayers = theano.function([index], self.getLayerOutput,
                   givens={self.x: train_set_x[index :]})
        
        def Layers():
            return getLayers(0)
        return pretrain_fns, Layers
    def build_finetune_functions(self,datasets,batch_size,learning_rate):
        
       
        (train_set_x,train_set_y) = datasets[0]
        (test_set_x,test_set_y) = datasets[1]
        
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size
        index = T.lscalar('index')
        gparams = T.grad(self.finetune_cost,self.params)
        updates = []
        for param,gparam in zip(self.params,gparams):
            updates.append((param,param-gparam*learning_rate))
        train_fn = theano.function([index],
                                   outputs=self.finetune_cost,
                                   updates=updates,
                                   givens={self.x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                           self.y: train_set_y[index * batch_size:(index + 1) * batch_size]})

        test_score_i = theano.function([index],self.errors,
                                       givens={self.x:test_set_x[index*batch_size:(index+1)*batch_size],
                                               self.y:test_set_y[index*batch_size:(index+1)*batch_size]})
        getlabel = theano.function([index],self.label,
                                   givens={self.x:test_set_x[index:],
                                           self.y:test_set_y[index:]})
        getfeature = theano.function([index],self.feature,
                                    givens={self.x:test_set_x[index:]})
        getLayers = theano.function([index], self.getLayerOutput,
                   givens={self.x: train_set_x[index :]})
        
        def Layers():
            return getLayers(0)
        def Features():
            return getfeature(0)
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]
        def get_test_label():
            return getlabel(0)
        return train_fn,test_score,get_test_label, Features, Layers
    
    def getParams(self,input):
        index = T.lscalar('index')
        getLayers = theano.function([index], self.getLayerOutput,
                   givens={self.x: input[index :]})
        
        def Layers():
            return getLayers(0)
        return Layers,self.sigmoid_layers[-1].output, self.params
#         return self.sigmoid_layers, self.params

    def save_params(self, fileName):
        save_file = open(fileName, 'wb')  # this will overwrite current contents
        print len(self.params),self.params
        for i in xrange(len(self.params)):
            if (i%2 ==0):
                W = self.params[i]
                cPickle.dump(W.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
            else:
                b = self.params[i]
                cPickle.dump(b.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
        save_file.close()
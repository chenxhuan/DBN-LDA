'''
Created on Sep 2, 2014

@author: chenxh
'''
import os, numpy, theano
import cPickle
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from src.preprocess.preprocess_data import *
from src.model.rbm import *

class Testing(object):
    
    def __init__(self,n_ins=1065,
                 hidden_layers_sizes=[100, 100, 100], n_outs=10, path = 'rightParameters'):
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layers_sizes)   # equal 2
        self.n_ins = n_ins
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_outs = n_outs
        assert self.n_layers > 0
        self.params = self.loadParams(path)
        print self.params
        
        # here why  ???
        numpy_rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented 
        self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels


        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
            # get W and b
            weight = self.params[i*2]
            biase =self.params[i*2+1]
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid, W=weight,b=biase)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)


            # Construct an RBM that shared weights with this layer
#             rbm_layer = RBM(numpy_rng=numpy_rng,
#                             theano_rng=theano_rng,
#                             input=layer_input,
#                             n_visible=input_size,
#                             n_hidden=hidden_layers_sizes[i],
#                             W=sigmoid_layer.W,
#                             hbias=sigmoid_layer.b)
#             self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs, W = self.params[2*self.n_layers], b = self.params[-1])

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
        self.label = self.logLayer.getLabel(self.y)
        self.outputFeature = self.logLayer.getFeature()
        self.getLayerOutput = self.sigmoid_layers[-1].Output()
#         self.predict = self.predictLabel(self.x,self.y)
    def loadParams(self,path):
        params = []
        save_file = open(path,'rb')
        W = theano.shared(value=numpy.zeros((self.n_ins,self.hidden_layers_sizes[0]),dtype=theano.config.floatX),name='W',borrow=True)
        b = theano.shared(value=numpy.zeros((self.hidden_layers_sizes[0],),dtype=theano.config.floatX),name='b',borrow=True)
        W.set_value(cPickle.load(save_file), borrow=True)
        b.set_value(cPickle.load(save_file), borrow=True)
        params.extend([W,b])
        for i in xrange(len(self.hidden_layers_sizes)-1):
            W = theano.shared(value=numpy.zeros((self.hidden_layers_sizes[i],self.hidden_layers_sizes[i+1]),dtype=theano.config.floatX),name='W',borrow=True)
            b = theano.shared(value=numpy.zeros((self.hidden_layers_sizes[i+1],),dtype=theano.config.floatX),name='b',borrow=True)
            W.set_value(cPickle.load(save_file), borrow=True)
            b.set_value(cPickle.load(save_file), borrow=True)
            params.extend([W,b])
        W = theano.shared(value=numpy.zeros((self.hidden_layers_sizes[-1],self.n_outs),dtype=theano.config.floatX),name='W',borrow=True)
        b = theano.shared(value=numpy.zeros((self.n_outs,),dtype=theano.config.floatX),name='b',borrow=True)
        W.set_value(cPickle.load(save_file), borrow=True)
        b.set_value(cPickle.load(save_file), borrow=True)
        params.extend([W,b])
        save_file.close()
        return params
        
    def predictLabel(self,input,y):
        
        index = T.lscalar('index')
#         test_score = theano.function([index], self.errors,
#                    givens={self.x: input[index :],
#                            self.y: y[index :]})
        getlabel = theano.function([index], self.label,
                  givens={self.x: input[index:],
                           self.y: y[index:]})
        getLayers = theano.function([index], self.getLayerOutput,
                   givens={self.x: input[index :]})
        
        getVectors = theano.function([index], self.outputFeature,
                   givens={self.x: input[index :]})
        
        def Layers():
            return getLayers(0)
#         def test_scores():
#             return test_score(0)
        def get_test_label():
            return getlabel(0)
        def outputVectors():
            return getVectors(0)
        
        return get_test_label,Layers,outputVectors
   
    
    
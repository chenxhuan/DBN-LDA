'''
Created on Oct 24, 2014

@author: chenxh
'''
import os, numpy, theano
import cPickle
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from preprocess_data import *

class ReC(object):
    
    def __init__(self,n_ins=1188,
                 hidden_layers_sizes=[100, 100, 100], n_outs=1188, top_params=None, params=None):
        self.sigmoid_layers = []
        self.n_layers = len(hidden_layers_sizes)   
        self.n_ins = n_ins
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_outs = n_outs
        assert self.n_layers > 0
        self.params = params
#         print self.params, top_params
        
        # here why  ???
        numpy_rng = numpy.random.RandomState(123)

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented 
        
        sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=self.x,
                                        n_in=self.n_ins,
                                        n_out=hidden_layers_sizes[-1],
                                        activation=T.nnet.sigmoid, W=(top_params[0][(n_ins-hidden_layers_sizes[-1]-1):-1]).T,b=self.params[-1])
        self.sigmoid_layers.append(sigmoid_layer)
        i=self.n_layers-1
        while (i>=0):
            if i == self.n_layers-1:
                input_size = hidden_layers_sizes[-1]
            else:
                input_size = hidden_layers_sizes[i]

            if i == self.n_layers-1:
                layer_input = sigmoid_layer.output
            else:
                layer_input = self.sigmoid_layers[-1].output
            # get W and b
            if(i>0):
                weight = self.params[i*2]
                biase =self.params[i*2-1]
                self.sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i-1],
                                        activation=T.nnet.sigmoid, W=weight.T,b=biase)

                # add the layer to our list of layers
                self.sigmoid_layers.append(self.sigmoid_layer)
            else:
                weight = self.params[0]
                self.sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.n_outs,
                                        activation=T.nnet.sigmoid, W=weight.T)
                self.sigmoid_layers.append(self.sigmoid_layer)
            i-=1
        print len(self.sigmoid_layers)
        self.getLayerOutput = self.sigmoid_layers[-1].Output()

       
    def getReconstructedOutput(self,input):
        index = T.lscalar('index')
        getLayers = theano.function([index], self.getLayerOutput,
                   givens={self.x: input[index :]})
        def Layers():
            return getLayers(0)
        return Layers
    
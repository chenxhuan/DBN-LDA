#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,numpy,theano,cPickle,sys,time, sklearn
import timeit

from src.experiment.evaluation import evaluation
from src.model.cnn import LeNetConvPoolLayer
from src.preprocess.preprocess_data import LogisticRegression, Rdata_load, change2PrimaryC, getAccuracy, HiddenLayer
# from mlp import HiddenLayer

sys.path.append("..")
import theano.tensor as T
# from evaluation import *
# from src.model.svmutil import *
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression as LR, Perceptron
# from sklearn import tree

__doc__ = 'CNN as the benchmark'
__author__ = 'Kangzhi Zhao'

def evaluate_lenet5(learning_rate=0.1, training_epochs=200,
                    dataset='annotation1000_20160927',
                    nkerns=[20, 50], batch_size=10, dataIndex=0):
    """ Demonstrates lenet on annotation1000 dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    filepath = '../../dataset/features/'+dataset
    saveFile = file("../../output/result_cnn.txt", 'a')
    # datasets = load_data(filepath)

    start_index = 0
    end_index = 116
    fold_size = 116
    (train_set_x, train_set_y, train_x, train_y), (test_set_x, test_set_y, test_x, test_y) \
        = Rdata_load(filepath, dataIndex*fold_size, (dataIndex + 1) * fold_size)

    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    # n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        inputs=[index],
        outputs=layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # validate_model = theano.function(
    #     [index],
    #     layer3.errors(y),
    #     givens={
    #         x: valid_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    # patience = 10000  # look as this many examples regardless
    # patience_increase = 2  # wait this much longer when a new best is
    #                        # found
    # improvement_threshold = 0.995  # a relative improvement of this much is
    #                                # considered significant
    # validation_frequency = min(n_train_batches, patience // 2)
    #                               # go through this many
    #                               # minibatche before checking the network
    #                               # on the validation set; in this case we
    #                               # check every epoch

    # best_validation_loss = numpy.inf
    # best_iter = 0
    # test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    # done_looping = False

    while epoch < training_epochs:
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            # if (iter + 1) % validation_frequency == 0:
            #
            #     # compute zero-one loss on validation set
            #     validation_losses = [validate_model(i) for i
            #                          in range(n_valid_batches)]
            #     this_validation_loss = numpy.mean(validation_losses)
            #     print('epoch %i, minibatch %i/%i, validation error %f %%' %
            #           (epoch, minibatch_index + 1, n_train_batches,
            #            this_validation_loss * 100.))
            #
            #     # if we got the best validation score until now
            #     if this_validation_loss < best_validation_loss:
            #
            #         #improve patience if loss improvement is good enough
            #         if this_validation_loss < best_validation_loss *  \
            #            improvement_threshold:
            #             patience = max(patience, iter * patience_increase)
            #
            #         # save best validation score and iteration number
            #         best_validation_loss = this_validation_loss
            #         best_iter = iter
            #
            #         # test it on the test set
            #         test_losses = [
            #             test_model(i)
            #             for i in range(n_test_batches)
            #         ]
            #         test_score = numpy.mean(test_losses)
            #         print(('     epoch %i, minibatch %i/%i, test error of '
            #                'best model %f %%') %
            #               (epoch, minibatch_index + 1, n_train_batches,
            #                test_score * 100.))
            #
            # if patience <= iter:
            #     done_looping = True
            #     break

    test = [
        test_model(i)
        for i in range(n_test_batches)
        ]
    predict_y = [t[0] for t in test]
    origin_y = [t[1] for t in test]
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    ev = evaluation(predict_y, origin_y)
    acc = getAccuracy(predict_y, origin_y)
    print 'results from  CNN , presion, recall, F1, accuracy: '
    print >> saveFile, 'first results from right DBN , precision, recall, F1, accuracy: '
    print >> saveFile, ev, acc
    print 'results from  DBN , presion, recall, F1, accuracy: '
    print ev, acc

    end_time = timeit.default_timer()
    print('Optimization complete.')
    # print('Best validation score of %f %% obtained at iteration %i, '
    #       'with test performance %f %%' %
    #       (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    # print(('The code for file ' +
    #        os.path.split(__file__)[1] +
    #        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
    evaluate_lenet5()
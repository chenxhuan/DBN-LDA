#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,numpy,sys,time
import timeit

from src.model.SdA import SdA
from src.experiment.evaluation import evaluation
from src.preprocess.preprocess_data import LogisticRegression, Rdata_load, change2PrimaryC, getAccuracy

sys.path.append("..")
import theano.tensor as T

__doc__ = 'CNN as the benchmark'
__author__ = 'Kangzhi Zhao'

def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=200,
             dataset='annotation1000_20160927', batch_size=1, dataIndex=0):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    n_out = 26
    # datasets = load_data(dataset)
    filepath = '../../dataset/features/'+dataset
    saveFile = file("../../output/result_sda.txt", 'a')

    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]
    # start_index = 0
    # end_index = 116
    fold_size = 100
    (train_set_x, train_set_y, train_x, train_y), (test_set_x, test_set_y, test_x, test_y) \
        = Rdata_load(filepath, dataIndex * fold_size, (dataIndex + 1) * fold_size)

    datasets = [0, 0, 0]
    datasets[0] = (train_set_x, train_set_y)
    # (valid_set_x, valid_set_y) = datasets[1]
    datasets[2] = (test_set_x, test_set_y)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    # n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=1188,
        hidden_layers_sizes=[594, 594, 594],
        n_outs=n_out
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))

    end_time = timeit.default_timer()

    print(('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    # # early-stopping parameters
    # patience = 10 * n_train_batches  # look as this many examples regardless
    # patience_increase = 2.  # wait this much longer when a new best is
    #                         # found
    # improvement_threshold = 0.995  # a relative improvement of this much is
    #                                # considered significant
    # validation_frequency = min(n_train_batches, patience // 2)
    #                               # go through this many
    #                               # minibatche before checking the network
    #                               # on the validation set; in this case we
    #                               # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    # done_looping = False
    epoch = 0

    while epoch < training_epochs:
        epoch = epoch + 1
        if epoch % 10 == 0:
            print('training @ epoch = ', epoch)
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            minibatch_avg_cost = train_fn(minibatch_index)
            # iter = (epoch - 1) * n_train_batches + minibatch_index
            #
            # if (iter + 1) % validation_frequency == 0:
            #     validation_losses = validate_model()
            #     this_validation_loss = numpy.mean(validation_losses)
            #     print('epoch %i, minibatch %i/%i, validation error %f %%' %
            #           (epoch, minibatch_index + 1, n_train_batches,
            #            this_validation_loss * 100.))
            #
            #     # if we got the best validation score until now
            #     if this_validation_loss < best_validation_loss:
            #
            #         #improve patience if loss improvement is good enough
            #         if (
            #             this_validation_loss < best_validation_loss *
            #             improvement_threshold
            #         ):
            #             patience = max(patience, iter * patience_increase)
            #
            #         # save best validation score and iteration number
            #         best_validation_loss = this_validation_loss
            #         best_iter = iter
            #
            #         # test it on the test set
            #         test_losses = test_model()
            #         test_score = numpy.mean(test_losses)
            #         print(('     epoch %i, minibatch %i/%i, test error of '
            #                'best model %f %%') %
            #               (epoch, minibatch_index + 1, n_train_batches,
            #                test_score * 100.))
            #
            # if patience <= iter:
            #     done_looping = True
            #     break

    end_time = timeit.default_timer()
    # print(
    #     (
    #         'Optimization complete with best validation score of %f %%, '
    #         'on iteration %i, '
    #         'with test performance %f %%'
    #     )
    #     % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    # )
    # print(('The training code for file ' +
    #        os.path.split(__file__)[1] +
    #        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
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
    # print 'results from sda , presion, recall, F1, accuracy: '
    print >> saveFile, 'first results from sda , precision, recall, F1, accuracy: '
    print >> saveFile, ev, acc
    print 'results from sda , presion, recall, F1, accuracy: '
    print ev, acc

    end_time = timeit.default_timer()
    print('Optimization complete.')


if __name__ == '__main__':
    for ind in range(10):
        print 'index @'+str(ind)
        test_SdA(dataIndex=ind)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,numpy,sys,time
import timeit
sys.path.append("..")
from src.model.SdA import SdA
from src.experiment.evaluation import evaluation
from src.preprocess.preprocess_data import *


import theano.tensor as T

__doc__ = 'CNN as the benchmark'
__author__ = 'Kangzhi Zhao'

def test_SdA(finetune_lr=0.1, pretraining_epochs=100,
             pretrain_lr=0.1, training_epochs=200,
             dataset='annotation1000_20160927', fold_size = 100, topicFile= 'lexicon2_20160928',hidden_layers=[594, 594, 594],
             n_ins= 1188, batch_size=10, dataIndex=0,lamda=0.05,
             corruption_levels = [0.05, 0.05, 0.05, 0.05,0.05]):
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
    sstart_time = timeit.default_timer()
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    n_out = 26
    # datasets = load_data(dataset)
    filepath = '../../dataset/features/'+dataset
    topicPath = '../../dataset/features/'+topicFile
    saveFile = file("../../output/result_sda_parameters.txt", 'a')
    print >> saveFile, 'round ', dataIndex, 'lamda ', lamda
    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]
    # start_index = 0
    # end_index = 116
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

    # build the topic set
    (topic_train, topic_label,topic_x,topic_y) = Ldata_load(topicPath)
    topic_list_set = []
    for y in train_y:
        topic_list_set.append(topic_x[y])
    topic_sets = theano.shared(numpy.asarray(topic_list_set,dtype=theano.config.floatX),borrow=True)


    print('... building the DSSDA model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_ins,
        hidden_layers_sizes=hidden_layers,
        n_outs=n_out,
        supervised_type=1
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x, supervised_set=topic_sets,
                                                batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr,
                                            ld=lamda))
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
    train_fn, test_model,test_labels = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    print('... finetunning the model')
    start_time = timeit.default_timer()
    # done_looping = False
    epoch = 0

    while epoch < training_epochs:
        epoch = epoch + 1
        c = []
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            minibatch_avg_cost = train_fn(minibatch_index)
            c.append(minibatch_avg_cost)
        print 'fine-tuning epoch %d, cost ' % epoch, numpy.mean(c)
    end_time = timeit.default_timer()
    predict_y, origin_y = test_labels()
    predict_y = change2PrimaryC(predict_y)
    origin_y = change2PrimaryC(origin_y)
    ev = evaluation(predict_y, origin_y)
    acc = getAccuracy(predict_y, origin_y)
    # print 'results from sda , presion, recall, F1, accuracy: '
    print >> saveFile,ev[0],'\t',ev[1],'\t',ev[2],'\t',acc
    print 'results from DSSDA , presion, recall, F1, accuracy: '
    print ev, acc
    end_time = timeit.default_timer()
    print 'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)


    # print('... building the LSSDA model')
    # # construct the stacked denoising autoencoder class
    # sda = SdA(
    #     numpy_rng=numpy_rng,
    #     n_ins=n_ins,
    #     hidden_layers_sizes=hidden_layers,
    #     n_outs=n_out,
    #     supervised_type=2
    # )
    # # end-snippet-3 start-snippet-4
    # #########################
    # # PRETRAINING THE MODEL #
    # #########################
    # print('... getting the pretraining functions')
    # pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x, supervised_set=train_set_y,
    #                                             batch_size=batch_size)
    #
    # print('... pre-training the model')
    # start_time = timeit.default_timer()
    # ## Pre-train layer-wise
    # for i in range(sda.n_layers):
    #     # go through pretraining epochs
    #     for epoch in range(pretraining_epochs):
    #         # go through the training set
    #         c = []
    #         for batch_index in range(n_train_batches):
    #             c.append(pretraining_fns[i](index=batch_index,
    #                                         corruption=corruption_levels[i],
    #                                         lr=pretrain_lr,
    #                                         ld=lamda))
    #         print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))
    # end_time = timeit.default_timer()
    # print(('The pretraining code for file ' +
    #        os.path.split(__file__)[1] +
    #        ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    # # end-snippet-4
    # ########################
    # # FINETUNING THE MODEL #
    # ########################
    # # get the training, validation and testing function for the model
    # print('... getting the finetuning functions')
    # train_fn, test_model,test_labels = sda.build_finetune_functions(
    #     datasets=datasets,
    #     batch_size=batch_size,
    #     learning_rate=finetune_lr
    # )
    # print('... finetunning the model')
    # start_time = timeit.default_timer()
    # # done_looping = False
    # epoch = 0
    #
    # while epoch < training_epochs:
    #     epoch = epoch + 1
    #     c = []
    #     for minibatch_index in range(n_train_batches):
    #         iter = (epoch - 1) * n_train_batches + minibatch_index
    #         minibatch_avg_cost = train_fn(minibatch_index)
    #         c.append(minibatch_avg_cost)
    #     print 'fine-tuning epoch %d, cost ' % epoch, numpy.mean(c)
    # end_time = timeit.default_timer()
    # predict_y, origin_y = test_labels()
    # predict_y = change2PrimaryC(predict_y)
    # origin_y = change2PrimaryC(origin_y)
    # ev = evaluation(predict_y, origin_y)
    # acc = getAccuracy(predict_y, origin_y)
    # # print 'results from sda , presion, recall, F1, accuracy: '
    # print >> saveFile,ev[0],'\t',ev[1],'\t',ev[2],'\t',acc
    # print 'results from LSSDA , presion, recall, F1, accuracy: '
    # print ev, acc
    # end_time = timeit.default_timer()
    # print 'Finish all using  %.2f mins' % ((end_time - start_time) / 60.)
    #
    # print('... building the SDA model')
    # # construct the stacked denoising autoencoder class
    # sda = SdA(
    #     numpy_rng=numpy_rng,
    #     n_ins=n_ins,
    #     hidden_layers_sizes=hidden_layers,
    #     n_outs=n_out
    # )
    # # end-snippet-3 start-snippet-4
    # #########################
    # # PRETRAINING THE MODEL #
    # #########################
    # print('... getting the pretraining functions')
    # pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
    #                                             batch_size=batch_size)
    #
    # print('... pre-training the model')
    # start_time = timeit.default_timer()
    # ## Pre-train layer-wise
    # for i in range(sda.n_layers):
    #     # go through pretraining epochs
    #     for epoch in range(pretraining_epochs):
    #         # go through the training set
    #         c = []
    #         for batch_index in range(n_train_batches):
    #             c.append(pretraining_fns[i](index=batch_index,
    #                                         corruption=corruption_levels[i],
    #                                         lr=pretrain_lr))
    #         print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))
    # end_time = timeit.default_timer()
    # print(('The pretraining code for file ' +
    #        os.path.split(__file__)[1] +
    #        ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    # # end-snippet-4
    # ########################
    # # FINETUNING THE MODEL #
    # ########################
    # # get the training, validation and testing function for the model
    # print('... getting the finetuning functions')
    # train_fn, test_model,test_labels = sda.build_finetune_functions(
    #     datasets=datasets,
    #     batch_size=batch_size,
    #     learning_rate=finetune_lr
    # )
    # print('... finetunning the model')
    # start_time = timeit.default_timer()
    # # done_looping = False
    # epoch = 0
    #
    # while epoch < training_epochs:
    #     epoch = epoch + 1
    #     c = []
    #     for minibatch_index in range(n_train_batches):
    #         iter = (epoch - 1) * n_train_batches + minibatch_index
    #         minibatch_avg_cost = train_fn(minibatch_index)
    #         c.append(minibatch_avg_cost)
    #     print 'fine-tuning epoch %d, cost ' % epoch, numpy.mean(c)
    # end_time = timeit.default_timer()
    # predict_y, origin_y = test_labels()
    # predict_y = change2PrimaryC(predict_y)
    # origin_y = change2PrimaryC(origin_y)
    # ev = evaluation(predict_y, origin_y)
    # acc = getAccuracy(predict_y, origin_y)
    # # print 'results from sda , presion, recall, F1, accuracy: '
    # print >> saveFile,ev[0],'\t',ev[1],'\t',ev[2],'\t',acc
    # print 'results from sda , presion, recall, F1, accuracy: '
    # print ev, acc
    end_time = timeit.default_timer()
    print 'Finish all using  %.2f mins' % ((end_time - sstart_time) / 60.)
    print('Optimization complete.')
    print >>saveFile,'------------------------------------------------------------------------------'



if __name__ == '__main__':
    # lds = [1,0.5,0.1,0.05,0.01,0.001,0]
    # for ld in lds:
    #     print 'ld ', ld
    #     test_SdA(dataIndex=9,lamda=ld)

    # test 4  corruption level
    # cls = [[0, 0, 0],[0.001, 0.001, 0.001],[0.01, 0.01, 0.01],[0.05, 0.05, 0.05],[0.1, 0.1, 0.1]]
    # for cl in cls:
    #     print 'cl ', cl
    #     test_SdA(corruption_levels=cl,dataIndex=9)

    # hls = [[100, 100, 100],[200, 200, 200],[300, 300, 300],[400, 400, 400],[500, 500, 500],[600, 600, 600],[700, 700, 700],[800, 800, 800],[900, 900, 900],[1000, 1000, 1000],[1100, 1100, 1100]]
    # for hl in hls:
    #     print 'hl ', hl
    #     test_SdA(hidden_layers=hl,dataIndex=9)

    # hls = [[594],[594, 594],[594, 594, 594],[594, 594, 594, 594],[594, 594, 594, 594, 594]]
    # for hl in hls:
    #     print 'hl ', hl
    #     test_SdA(hidden_layers=hl,dataIndex=9)

    #  test different vectors
    for ind in range(10):
        print 'index @'+str(ind)
    # test_SdA(dataset='annotation1000_20160927_doc2vec', n_ins=1188, topicFile='lexicon2_20160928_doc2vec',dataIndex=9)
        test_SdA(dataset='annotation1000_20160927bow', n_ins=1065,topicFile= 'lexicon2_20160928bow',dataIndex=ind)
        test_SdA(dataset='annotation1000_20160927tfidf', topicFile= 'lexicon2_20160928tfidf',dataIndex=ind)
    # test_SdA(dataset='mixed_5_20161007tfidf', topicFile= 'lexicon2_20160928tfidf')
        test_SdA(dataIndex=ind)

    # for ind in range(2):
    #     print 'index @'+str(ind+8)
    #     test_SdA(dataIndex=ind+8)

#encoding=utf-8
#!/usr/bin/python
'''
Created on 2016-11-6

@author: Kangzhi
'''

import sys
import numpy as np
import gensim
import time
from random import shuffle

from gensim.models.doc2vec import Doc2Vec, LabeledSentence

LabeledSentence = gensim.models.doc2vec.LabeledSentence

def get_dataset(inputFile):
    #读取数据
    with open(inputFile, 'r') as infile:
        texts = []
        lables = []
        for line in infile.readlines():
            if len(line.split('\t')) != 2:
                print "wrong line"
                print line
                continue
            label, text = line.split("\t")
            texts.append(text)
            lables.append(label)

    #Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
    #我们使用Gensim自带的LabeledSentence方法. 标识的格式为"DATA_i", 其中i为序号
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    texts = labelizeReviews(texts, 'DATA')

    return texts, lables

##对数据进行训练
def train(texts,size = 400,epoch_num=10):
    #实例DM和DBOW模型
    # model_dm = gensim.models.Doc2Vec(min_count=1, window=15, size=size, sample=1e-4, negative=5, workers=25)
    # model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=25)
    model_dm = gensim.models.Doc2Vec(workers=25)

    #使用所有的数据建立词典
    model_dm.build_vocab(texts)
    # model_dbow.build_vocab(texts)

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    temp = texts[:]
    # print texts
    # all_reviews = np.asarray(texts, dtype=object)
    # all_reviews = texts
    for epoch in range(epoch_num):
        print "epoch %d in epoch_num %d" % (epoch, epoch_num)
        # perm = np.random.permutation(all_reviews.shape[0])
        shuffle(temp)
        model_dm.train(temp)
        # model_dbow.train(texts)

    # return model_dm, model_dbow
    return model_dm


##读取向量
def getVecs(model, corpus, size=300):
    vecs = []
    for z in corpus:
        print z.tags
        vecs.append(np.array(model.docvecs[z.tags[0]]).reshape((1, size)))
    return np.concatenate(vecs)


##将训练完成的数据转换为vectors
def get_vectors(model_dm):
    #获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, texts, size=300)
    # train_vecs_dbow = getVecs(model_dbow, texts, size)
    # train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    # return train_vecs
    return train_vecs_dm


def writefile(filepath, labels, features):
    if len(labels) != features.shape[0]:
        print "label unmatched"
        print "labels:"+str(len(labels))
        print "features:"+str(len(features.shape[0]))
        return False
    with open(filepath, 'w') as f:
        for id in range(len(labels)):
            string = ''+str(labels[id])+'\t'
            string += ' '.join(str(x) for x in features[id])
            f.write(string+'\n')


if __name__ == "__main__":
    # size, epoch_num = 1188/2, 10
    # files = ['annotation1000_20160927', 'mixed_5_20161007', 'tnbz_20161007', 'zzcxhg_20161007']
    files = ['lexicon2_20160928']
    for filename in files:
        # filename = 'test'
        print "now applying doc2vec "+filename
        start = time.time()
        #设置向量维度和训练次数
        #In the paper, the author generate the doc vector by concatenate two model vectors
        inputFile = '../../dataset/wordSegmentation/'+filename
        saveFile = '../../dataset/features/'+filename+'_doc2vec'
        #获取数据及其类别标注
        texts, labels = get_dataset(inputFile)
        # print texts
        #对数据进行训练，获得模型
        print "training..."
        model_dm = train(texts, epoch_num=50)
        #从模型中抽取文档相应的向量
        print "training finished"
        # vecs = get_vectors(model_dm)
        vecs = getVecs(model_dm, texts)
        writefile(saveFile, labels, vecs)
        end = time.time()
        print filename+" success with %ds" %(end-start)

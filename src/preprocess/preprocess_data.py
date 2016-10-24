#encoding=utf-8
'''
Created on 2014-3-14

@author: xinhuan
'''
from numpy import *
import os , cPickle, theano, numpy
import theano.tensor as T
from scipy import *

def Rdata_load(dataset, fi = None, ti = None, columnFi = None,columnTi=None):
#     data_dir, data_file = os.path.split(dataset)
#     if data_dir =='' or not os.path.isfile(dataset):
#         ''' __file__ 是用来获得模块所在的路径 '''
#         new_path = os.path.join(os.path.split(__file__)[0],"..","data",dataset)
#         if os.path.isfile(new_path) :
#             dataset = new_path
    print '... loading right data'
    f = open(dataset,'rb')
    lines = f.readlines()
    lenLine = len(lines)
    print lenLine
    data_x = []
    label = []
    x = []
    y =[]
    index = 0
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        if len(temp) < 2:
            continue
        dataArray = []
        for val in temp[1].split(' '):
            dataArray.append(float(val))
        data_x.append(array(dataArray))
        label.append(int(temp[0]))
#     print shape(label),label
    def shared_dataset(data_x, data_y,borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x, T.cast(shared_y,'int32')
    test_set_x,test_set_y = shared_dataset([data_x[i+fi][columnFi:columnTi] for i in xrange(ti-fi)],label[fi:ti]) 
#     test_set_x,test_set_y = shared_dataset(data_x,label) 
    x+=(([data_x[i][columnFi:columnTi] for i in xrange(fi)])) 
    x+=(([data_x[i+ti][columnFi:columnTi] for i in xrange(len(data_x)-ti)]))
    y+=((label[0:fi]))
    y+=((label[ti:]))
#     x+=((data_x)) 
#     y+=((label))
    print shape(x),shape(y)
    train_set_x, train_set_y = shared_dataset(x,y)
    rval = [(train_set_x,train_set_y,x,y),(test_set_x,test_set_y,data_x[fi:ti],label[fi:ti])]
    return rval

def Ldata_load(dataset, fi = 0, ti = 0):
#     data_dir, data_file = os.path.split(dataset)
#     if data_dir =='' or not os.path.isfile(dataset):
#         ''' __file__ 是用来获得模块所在的路径 '''
#         new_path = os.path.join(os.path.split(__file__)[0],"..","data",dataset)
#         if os.path.isfile(new_path):
#             dataset = new_path
    print '... loading left data'
    f = open(dataset,'rb')
    lines = f.readlines()
    lenLine = len(lines)
    print lenLine
    data_x = []
    label = []
    x = []
    y =[]
    index = 0
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        if len(temp) < 2:
            continue
        dataArray = []
        for val in temp[1].split(' '):
            dataArray.append(float(val))
        data_x.append(array(dataArray))
        label.append(int(temp[0]))
#     for line in lines:
#         line = line.strip()
#         temp = line.split('\t')
#         if len(temp) < 1:
#             continue
#         dataArray = []
#         for val in temp:
#             dataArray.append(float(val))
#         data_x.append(array(dataArray))
#         label.append(int(index))
#         index +=1
    print shape(data_x)
    def shared_dataset(data_x, data_y,borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x, T.cast(shared_y,'int32')
    x+=((data_x[0:fi])) 
    x+=((data_x[ti:]))
    y+=((label[0:fi]))
    y+=((label[ti:]))
    train_set_x, train_set_y = shared_dataset(x,y)
    rval = [train_set_x, train_set_y,x,y]
    return rval

def data_load(dataset, fi = None, ti = None):
    data_dir, data_file = os.path.split(dataset)
    if data_dir =='' or not os.path.isfile(dataset):
        ''' __file__ 是用来获得模块所在的路径 '''
        new_path = os.path.join(os.path.split(__file__)[0],"..","data",dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl':
            dataset = new_path
    print '... loading data'
    f = open(dataset,'rb')
    lines = f.readlines()
    lenLine = len(lines)
    data_x = []
    label = []
    x = []
    y =[]
    index = 0
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        if len(temp) < 2:
            continue
        dataArray = []
        for val in temp[1].split(' '):
            dataArray.append(float(val))
        data_x.append(array(dataArray))
        label.append(int(temp[0]))
    print shape(data_x)
#     print shape(label),label
    def shared_dataset(data_x, data_y,borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x, T.cast(shared_y,'int32')
#     data_set_x, data_set_y = shared_dataset(data_x[fi:ti],label[fi:ti])
    test_set_x, test_set_y = shared_dataset(data_x[fi:ti],label[fi:ti]) 
    valid_set_x,valid_set_y = shared_dataset(data_x[fi:ti],label[fi:ti])
    x+=((data_x[0:fi])) 
    x+=((data_x[ti:]))
    y+=((label[0:fi]))
    y+=((label[ti:]))
    train_set_x, train_set_y = shared_dataset(x,y)
    rval = [(train_set_x, train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)]
    return rval
#     return data_set_x, data_set_y
        
    

def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir =='' or not os.path.isfile(dataset):
        ''' __file__ 是用来获得模块所在的路径 '''
        new_path = os.path.join(os.path.split(__file__)[0],"..","data",dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl':
            dataset = new_path
    print '... loading data'
    f = open(dataset,'rb')
    """ 使用load()函数从文件中取出已保存的对象时，pickle知道如何恢复这些对象到它们本来的格式"""
    train_set, valid_set, test_set = cPickle.load(f)
    f.close
    fil = file("tttt.txt",'a')
    # if Theano is using a GPU device, then the borrow flag has no effect
    def shared_dataset(data_xy,borrow=True):
        data_x, data_y = data_xy
#         for a in data_x[:500]:
#             print >>fil, '%s ' % a[:]
#         fil.flush()
        shared_x = theano.shared(numpy.asarray(data_x[:500],dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y[:500],dtype=theano.config.floatX),borrow=borrow)
        return shared_x, T.cast(shared_y,'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)   
    valid_set_x,valid_set_y = shared_dataset(valid_set)
    train_set_x,train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)]
    return rval


class HiddenLayer(object):
    def __init__(self,rng, input, n_in, n_out, W=None, b=None,activation = T.tanh):
        self.input = input
        
        if W is None:
            W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(n_in+n_out)),high=numpy.sqrt(6./(n_in+n_out)),
                                                 size=(n_in,n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *=4
            W = theano.shared(value=W_values, name='W',borrow=True)
            
        if b is None:
            b_values = numpy.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(value= b_values, name='b', borrow = True)
        self.W = W
        self.b = b
        lin_output = T.dot(input,self.W)+self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W,self.b]
    def Output(self):
        return self.output 
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out,W=None, b=None):
        if W is None:
            W = theano.shared(value=numpy.zeros((n_in,n_out),dtype=theano.config.floatX),name='W',borrow=True)
        if b is None:
            b = theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
        self.W = W
        self.b = b
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
        self.params = [self.W, self.b]
        
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self,y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return self.y_pred, y, T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError() 
    def getLabel(self,y):
        return  self.y_pred, y  
    def getPredict(self,input):
        return  self.y_pred
    def getFeature(self):
        return self.p_y_given_x    

def similarity(r_layer,l_layer):
#     if len(r_layer[0]) != len(l_layer[0]):
#         raise TypeError('right should have the same shape as left',
#                 ('r', r_layer[0].type, 'l', l_layer[0].type))
#     print shape(r_layer),shape(l_layer)
    r_sizes = len(r_layer)
    l_sizes = len(l_layer)
    out_result = []
    for rdata in xrange(r_sizes):
        p_simi = []
        for ldata in xrange(l_sizes):
#             p = dot(r_layer[rdata],l_layer[ldata])/(linalg.norm(r_layer[rdata])*linalg.norm(l_layer[ldata]))
            p = PearsonCorrelation(r_layer[rdata],l_layer[ldata])
#             p = rootMeanSquareError(r_layer[rdata],l_layer[ldata])
            p_simi.append(p)
        y_MaxSim = argmax(p_simi)
        out_result.append(y_MaxSim)
    return out_result
def distance(r_layer,l_layer):
    r_sizes = len(r_layer)
    l_sizes = len(l_layer)
    out_result = []
    for rdata in xrange(r_sizes):
        p_simi = []
        for ldata in xrange(l_sizes):
            p = rootMeanSquareError(r_layer[rdata],l_layer[ldata])
            p_simi.append(p)
        y_minSim = argmin(p_simi)
        out_result.append(y_minSim)
    return out_result
def distance1(vectors,size = 40):
    out_result = []
    p_distance = []
    j = 0
    for i in xrange(len(vectors)):
        dis = linalg.norm(vectors[i])
        p_distance.append(dis)
        j += 1
        if(j>=size):
            out_result.append(argmin(p_distance)) 
            p_distance = []
            j = 0
    return out_result
def cal_sim(a, b):
    if len(a) !=len(b):
        print "length error."
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1,b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down ==0.0:
        print "down is 0,error."
        return None
    else:
        return part_up/part_down     
def rootMeanSquareError(vector1,vector2):
    ret = sqrt(((array(vector1)-array(vector2))**2).mean())
    return ret
def PearsonCorrelation(vector1,vector2):
    ret = (cov(vector1,vector2)[0][1])/(sqrt(cov(vector1)*cov(vector2)))
    return ret
def getAccuracy(v1,v2):
    total_correct = 0
    for v, y in zip(v1, v2):
        if y == v: 
            total_correct += 1
    return 1.0*total_correct/len(v1)

def change2PrimaryC(vector):
    output =[]
    l = len(vector)
    for i in xrange(l):
        f = vector[i]
        if(f==1 or f==2 or f==24 or f==25):
            output.append(0)
        if(f==0 or f==11 or f==23):
            output.append(1)
        if(f==3 or f==4 or f==5 or f==6 or f==12 or f==13 or f==14 or f==15 or f==17):
            output.append(2)
        if(f==7 or f==9 or f==16 or f==18 or f==20 or f==21):
            output.append(3)
        if(f==10):
            output.append(4)
        if(f==8 or f==19 or f==22):
            output.append(5) 
    return output

# print change2PrimaryC([8,21,8,22,20,20,20,3,20,16,2,20,15,20,20,9,3,21,5,8,22,21,9,22,21
# ,3,21,23,7,20,3,9,9,17,20,9,2,21,13,20,21,15,20,15,20,5,20,20,16,9
# ,8,20,3,20,9,20,3,9,8,20,21,3,2,9,9,21,9,1,14,13,9,17,1,2,9
# ,5,18,3,15,13,9,15,1,21,9,20,11,21,20,9,20,0,9,8,0,21,21,13,2,21])

# def pearson(x,y):
#     n=len(x)
#     vals=range(n)
# # Simple sums
#     sumx=sum([float(x[i]) for i in vals])
#     sumy=sum([float(y[i]) for i in vals])
# # Sum up the squares
#     sumxSq=sum([x[i]**2.0 for i in vals])
#     sumySq=sum([y[i]**2.0 for i in vals])
# # Sum up the products
#     pSum=sum([x[i]*y[i] for i in vals])
# # Calculate Pearson score
#     num=pSum-(sumx*sumy/n)
#     den=((sumxSq-pow(sumx,2)/n)*(sumySq-pow(sumy,2)/n))**.5
#     if den==0: return 0
#     r=num/den
#     return r


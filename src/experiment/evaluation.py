'''
Created on 2014-4-22

@author: xinhuan
'''
import numpy


def evaluation(predict_y,origin_y):
    predictDirR = {}
    predictDir = {}
    orginDirP = {}
    orginDir = {}
    for ii in range(len(predict_y)):
        y = predict_y[ii]
        if predict_y[ii]==origin_y[ii]:
            if predictDirR.has_key(y):
                predictDirR[y]=predictDirR.get(y)+1
            else :
                predictDirR[y]=1
    for y in origin_y:
        if orginDir.has_key(y):
            orginDir[y]=orginDir.get(y)+1
        else :
            orginDir[y]=1
    for ii in range(len(origin_y)):
        y = origin_y[ii]
        if predict_y[ii]==origin_y[ii]:
            if orginDirP.has_key(y):
                orginDirP[y]=orginDirP.get(y)+1
            else :
                orginDirP[y]=1        
    for y in predict_y:
        if predictDir.has_key(y):
            predictDir[y]=predictDir.get(y)+1
        else :
            predictDir[y]=1
#     print predictDir, orginDir
    precision = []
    # calculate precision rate
    for key in predictDir.keys():
        if orginDirP.has_key(key):
            precision.append((orginDirP.get(key)*1.0)/predictDir.get(key))
        else:
            precision.append(0)
    test_precision = numpy.mean(precision)
    recalls = []
    # calculate recall rate
    for key in orginDir.keys():
        if predictDirR.has_key(key):
            recalls.append((predictDirR.get(key)*1.0)/orginDir.get(key))
        else:
            recalls.append(0)
    test_recall = numpy.mean(recalls)
    F1 = test_precision*test_recall*2/(test_precision+test_recall)
    return [test_precision,test_recall,F1]

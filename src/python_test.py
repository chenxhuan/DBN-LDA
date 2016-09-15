from src.model.svmutil import *

filepath = "../dataset/features/zzcxhg_feature_2014-11-16"
saveFile = file("../output/result2.txt",'a')

filepath1=unicode(filepath,'utf8')
y, x = svm_read_dataset(filepath1)
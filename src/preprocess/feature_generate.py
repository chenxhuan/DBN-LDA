#encoding=utf-8
'''
Created on 2016-9-26

@author: xinhuan
'''
import codecs,sys, os, time
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append("..")
reload(sys)

# 添加用户词典列表
userDics = []
input_userDic = codecs.open("../../dataset/user.dic","r","utf-8")
userDic = input_userDic.readlines()
for word in userDic:
    word = word.strip().replace("\n","").replace("\r","")
    userDics.append(word)
input_userDic.close()

def getDicFeatures(str):
    tmp = ''
    for w in userDics:
        if w in str:
            tmp += '1.0 '
        else:
            tmp += '0.0 '
    return tmp
def process(inputFile, saveFile, topN):
    input = codecs.open(inputFile, 'r', 'utf-8')
    output = codecs.open(saveFile, 'w', 'utf-8')
    lines = input.readlines()
    binaryFL = []
    wordVec = []
    for line in lines:
        if len(line.split('\t')) < 2:
            continue
        binaryF = ''
        label  = line.split('\t')[0]
        segs = line.split('\t')[1].strip()
        binaryF += label + '\t' + getDicFeatures(segs)
        binaryFL.append(binaryF)
        wordVec.append(segs)
    vectorizer=CountVectorizer()  #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i样例下的词频
    transformer=TfidfTransformer()   #该样例会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(vectorizer.fit_transform(wordVec))  #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()  #获取词袋模型中的所有词语
    weight=tfidf.toarray()  #将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i样例中的tf-idf权重
    tfidfMean = np.mean(weight, axis=0)
    wordMap = {}
    for i in xrange(len(word)):
        wordMap[i] = tfidfMean[i]
    wordList = sorted(wordMap.iteritems(), key=lambda asd:asd[1], reverse=True)[:topN]
    print len(wordList), len(word)
    for i in xrange(len(binaryFL)):
        tmpStr = binaryFL[i]
        for k,v in wordList:
            # tmpStr += weight[i][k] + ' '
            tmpStr += str(weight[i][k]) + ' '
        tmpStr = tmpStr.strip()+'\n'
        output.write(tmpStr)
    input.close()
    output.close()



if __name__ == "__main__":
    if len(sys.argv) != 3:
        start = time.time()
        # date = time.strftime('%Y%m%d',time.localtime(time.time()))
        inputFile = '../../dataset/wordSegmentation/zzcxhg_20161007'
        saveFile = '../../dataset/features/'+inputFile.split('/')[-1].split('.txt')[0]
        process(inputFile, saveFile,123)
        end = time.time()
        print 'finish :', end - start, 's'
    else:
        start = time.time()
        # date = time.strftime('%Y%m%d',time.localtime(time.time()))
        inputFile = '../../dataset/wordSegmentation/'+sys.argv[1]
        saveFile = '../../dataset/features/'+inputFile.split('/')[-1].split('.txt')[0]
        process(inputFile, saveFile, int(sys.argv[2]))
        end = time.time()
        print 'finish:', end - start, 's'
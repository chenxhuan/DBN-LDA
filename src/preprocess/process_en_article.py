#encoding=utf-8
import jieba, codecs, re, sys, os, time
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append("..")
reload(sys)

#去停词
r = "[0-9\[+.!/_,$%^*(+\"\']+|[+——！,﹌“·”《’‘》;…:：；℅～~@#￥%……&*】【？?\n\t]+"

def getSample(inputFile, saveFile):
    input = codecs.open(inputFile, 'r', 'utf-8')
    output = codecs.open(saveFile, 'w', 'utf-8')
    headLine = input.readline()
    lines = input.readlines()
    num1, num2, num3 = 0, 0, 0
    nums = 310
    for line in lines:
        segs = line.replace('\"','').split('\t')
        tmp = re.sub(r.decode('utf8'), u' '.decode('utf8'), segs[-1]).strip()
        if(len(segs) < 10 or segs[1] == '' or tmp == ''):
            continue
        # print segs[1], tmp
        if(segs[1] == 'Gestational-Diabetes' and num1 < nums):
            output.write('0\t'+tmp+'\n')
            num1 += 1
        elif(segs[1] == 'Adults-Living-with-Type-1' and num2 < nums):
            output.write('1\t'+tmp+'\n')
            num2 += 1
        elif(segs[1] == 'Adults-Living-with-Type-2' and num3 < nums):
            output.write('2\t'+tmp+'\n')
            num3 += 1
    input.close()
    output.close()
def getFeature(inputFile, saveFile, topN):
    input = codecs.open(inputFile, 'r', 'utf-8')
    output = codecs.open(saveFile, 'w', 'utf-8')
    lines = input.readlines()
    wordVec = []
    labels = []
    for line in lines:
        if len(line.split('\t')) < 2:
            continue
        label = line.split('\t')[0]
        segs = line.split('\t')[1].strip()
        labels.append(label)
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
    for i in xrange(len(labels)):
        tmpStr = labels[i]+'\t'
        for k,v in wordList:
            tmpStr += str(weight[i][k]) + ' '
        tmpStr = tmpStr.strip()+'\n'
        output.write(tmpStr)
    input.close()
    output.close()



if __name__ == "__main__":
    if len(sys.argv) != 2:
        start = time.time()
        date = time.strftime('%Y%m%d',time.localtime(time.time()))
        # inputFile = '../../dataset/originalData/ada.dsv'
        # saveFile = '../../dataset/wordSegmentation/'+inputFile.split('/')[-1].split('.dsv')[0]+ '_'+date
        # getSample(inputFile, saveFile)
        inputFile = '../../dataset/wordSegmentation/ada_20161210'
        saveFile = '../../dataset/features/'+inputFile.split('/')[-1].split('.txt')[0] + '_tfidf'
        getFeature(inputFile, saveFile,1188)
        end = time.time()
        print 'finish ws :', end - start, 's'
    else:
        print sys.argv[1]
        start = time.time()
        date = time.strftime('%Y%m%d',time.localtime(time.time()))
        inputFile = sys.argv[1]
        saveFile = '../../dataset/wordSegmentation/'+inputFile.split('/')[-1].split('.dsv')[0]+ '_'+date
        getSample(inputFile, saveFile)
        end = time.time()
        print 'finish ws :', end - start, 's'
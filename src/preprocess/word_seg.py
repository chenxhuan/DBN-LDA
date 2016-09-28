#encoding=utf-8
'''
Created on 2016-9-26

@author: xinhuan
'''
import jieba, codecs, re, sys, os, time
sys.path.append("..")
reload(sys)

# 添加用户词典
input_userDic = codecs.open("../../dataset/user.dic","r","utf-8")
userDic = input_userDic.readlines()
for word in userDic:
    word = word.strip().replace("\n","").replace("\r","").encode("utf-8")
    jieba.add_word(word)
input_userDic.close()

#去停词
r = "[A-Za-z0-9\[+.!/_,$%^*(+\"\']+|[+——！，,。﹌“·”《’‘》;…:：；℅～、~@#￥%……&*（）】【？?\n\t]+"
noUsedWords = {}
input_stopwords = codecs.open('../../dataset/stopword-full.dic','r','utf-8')
i = 0
for word in input_stopwords.readlines():
    word = word.strip().replace("\n","").replace("\r","").encode("utf-8")
    i += 1
    noUsedWords[str(word)] = i
input_stopwords.close()

#get the labels
labels = {}
input_topics = codecs.open('../../dataset/topics','r','utf-8')
i = 0
for word in input_topics.readlines():
    word = word.strip().replace("\n","").replace("\r","").encode("utf-8")
    labels[str(word)] = str(i)
    i += 1
input_topics.close()

def process(inputFile, saveFile):
    input = codecs.open(inputFile, 'r', 'utf-8')
    output = codecs.open(saveFile, 'w', 'utf-8')
    lines = input.readlines()
    for line in lines:
        res_line = ''
        tmp = line.split('\t')
        labelNo = labels[tmp[0].strip().split(';')[0].encode("utf-8")]
        res_line += labelNo + '\t'
        segs = jieba.cut(re.sub(r.decode('utf8'), u' '.decode('utf8'), tmp[1]).strip())
        for seg in segs:
            seg = seg.strip()
            if not noUsedWords.has_key(seg):
                res_line += seg + ' '
        res_line = res_line.strip()+ '\n'
        output.write(res_line)
    output.close()
    input.close()

if __name__ == "__main__":
    print len(sys.argv)
    if len(sys.argv) != 2:
        start = time.time()
        date = time.strftime('%Y%m%d',time.localtime(time.time()))
        inputFile = '../../dataset/originalData/annotation1000.txt'
        saveFile = '../../dataset/wordSegmentation/'+inputFile.split('/')[-1].split('.txt')[0]+ '_'+date
        process(inputFile, saveFile)
        end = time.time()
        print 'finish ws :', end - start, 's'
    else:
        print sys.argv[1]
        start = time.time()
        date = time.strftime('%Y%m%d',time.localtime(time.time()))
        inputFile = sys.argv[1]
        saveFile = '../../dataset/wordSegmentation/'+inputFile.split('/')[-1].split('.txt')[0]+ '_'+date
        process(inputFile, saveFile)
        end = time.time()
        print 'finish ws :', end - start, 's'
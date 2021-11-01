'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *
import numpy as np

def loadDataSet():
    """
    创建实验样本
    :return: 实验样本切分的词条 类别标签向量
    """
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签 1：侮辱性词汇 0：不是
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList, classVec
                 
def createVocabList(dataSet):
    """
    将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    :param dataSet:
    :return:
    """
    # 创建一个空的不重复列表
    vocabSet = set([])  #create empty set
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    更加vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
    :param vocabList: 词汇表
    :param inputSet: 数据
    :return:
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0]*len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        if word in vocabList:
            # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    :param trainCategory: 训练类别标签向量，即loadDataSet返回的classVec
    :return:
        p0Vect：侮辱类的条件概率数组
        p1Vect：非侮辱类的条件概率数组
        pAbusive：文档属于侮辱类的概率
    """
    # 计算训练的文档数目
    numTrainDocs = len(trainMatrix)
    # 计算每篇文章的词条数
    numWords = len(trainMatrix[0])
    # 文档属于侮辱类的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 创建numpy.ones数组，词条初始化为1，拉普拉斯平滑
    p0Num = ones(numWords)
    p1Num = ones(numWords)      #change to ones()
    # 分母初始化为2.0，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 相除
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    # 返回属于侮辱类的条件概率
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类器分类函数
    :param vec2Classify: 待分类的词条数组
    :param p0Vec: 侮辱类的条件概率数组
    :param p1Vec: 非侮辱类的条件概率数组
    :param pClass1: 文档属于侮辱类的概率
    :return:
    """
    # 对应元素相乘,logA*B=logA+logB，所以要加上np.log(pClass1)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    """
    测试朴素贝叶斯分类器
    :return:
    """
    # 创建实验样本
    listOPosts,listClasses = loadDataSet()
    print('listOPosts:\n', listOPosts)
    print('listClasses:\n', listClasses)
    # 创建词汇表
    myVocabList = createVocabList(listOPosts)
    print('myVocabList:\n', myVocabList)
    # 测试矩阵
    trainMat=[]
    for postinDoc in listOPosts:
        # 将实验样本向量化
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print('trainMat:\n', trainMat)
    # 训练朴素贝叶斯分类器
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('pAb:\n', pAb)
    # 测试样本1
    testEntry = ['love', 'my', 'dalmation']
    # 测试样本向量化
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('thisDoc:\n', thisDoc)
    # 分类
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    # 测试样本2
    testEntry = ['stupid', 'garbage']
    # 测试样本向量化
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('thisDoc:\n', thisDoc)
    # 分类
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):    #input is big string, #output is word list
    """
    接收一个大字符串并将其解析为字符串列表
    :param bigString:
    :return:
    """
    # print(f"bigString:\n{bigString}")
    import re
    # listOfTokens = re.split(r'\W*', bigString)
    listOfTokens = re.split(r'[-_.,!@#$%^&*()? \n~/]', bigString)
    # print(f"listOfTokens:\n{listOfTokens}")
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    # 遍历25个txt文件
    for i in range(1,26):
        # 读取每个垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('./Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        # 标记垃圾邮件，1表示垃圾文件
        classList.append(1)
        # 读取每个非垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('./Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        # 标记非垃圾邮件，1表示垃圾文件
        classList.append(0)
    print(f"docList:\n{docList}")
    print(f"fullText:\n{fullText}")
    # 创建词汇表，不重复
    vocabList = createVocabList(docList)#create vocabulary

    # 创建存储训练集的索引值的列表和测试集的索引值的列表
    trainingSet = list(range(50))
    testSet=[]           #create test set
    # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
    for i in range(10):
        # 随机选取索索引值
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        # 在训练集列表中删除添加到测试集的索引值
        del (trainingSet[randIndex])
    print(f"trainingSet:\n{trainingSet}")
    print(f"testSet:\n{testSet}")

    # 创建训练集矩阵和训练集类别标签系向量
    trainMat=[]
    trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 将生成的词集模型添加到训练矩阵中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 将类别添加到训练集类别标签系向量中
        trainClasses.append(classList[docIndex])
    # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    # 错误分类计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:        #classify the remaining items
        # 测试集的词集模型
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 如果分类错误，错误计数加1
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

if __name__=='__main__':
    # postingList, classVec = loadDataSet()
    # print('postingList:\n', postingList)
    # myVocabList = createVocabList(postingList)
    # print('myVocabList:\n', myVocabList)
    # trainMat = []
    # for postinDoc in postingList:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print('trainMat:\n', trainMat)
    # print('classVec:\n', classVec)
    # p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # print('p0V:\n', p0V)
    # print('p1V:\n', p1V)
    # print('classVec:\n', classVec)
    # print('pAb:\n', pAb)

    # testingNB()

    spamTest()
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def creatDataSet():
    # 数据集
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集和分类属性
    return dataSet, labels

def calcShannonEnt(dataSet):
    """
    计算给定数据集的经验熵（香农熵）
    :param dataSet: 数据集
    :return: 经验熵
    """
    # 返回数据集行数
    numEntries = len(dataSet)
    # 保存每个标签（label）出现次数的字典
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 提取标签信息
        if currentLabel not in labelCounts.keys():  # 如果标签没有放入统计次数的字典，添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # label计数

    shannonEnt = 0.0  # 经验熵
    # 计算经验熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 选择该标签的概率
        shannonEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannonEnt  # 返回经验熵
    
def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet:待划分的数据集
    :param axis:划分数据集的特征
    :param value:需要返回的特征的值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    """
    计算给定数据集的经验熵（香农熵）
    :param dataSet:
    :return:
    """
    # 特征数量
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    # 计数数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):        #iterate over all the features
        # 获取dataset的第i个所有特征
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # 创建set集合{}，元素不可重复
        uniqueVals = set(featList)       #get a set of unique values
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt((subDataSet))
        # 信息增益
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        # 打印每个特征的信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        # 计算信息增益
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            # 更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            # 记录信息增益最大的特征的索引值
            bestFeature = i
            # 返回信息增益最大特征的索引值
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    """
    统计classList中出现次数最多的元素（类标签）
    :param classList:类标签列表
    :return:
    """
    classCount={}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # 根据字典的值降序排列
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    """
    创建决策树
    :param dataSet: 训练数据集
    :param labels: 分类属性标签
    :return:
    """
    # 取分类标签（是否放贷：yes or no）
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同，则停止继续划分
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # 选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最有特征的标签
    bestFeatLabel = labels[bestFeat]
    # 根据最优特征生成树
    myTree = {bestFeatLabel:{}}
    # 删除已经使用的标签
    del(labels[bestFeat])
    # 得到最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 去掉重复的属性值
    uniqueVals = set(featValues)
    # 遍历特征,生成特证树
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
#main函数
if __name__=='__main__':
    dataSet, features = creatDataSet()
    # # print(dataSet)
    # # print(calcShannonEnt(dataSet))
    # print("最优索引值：" + str(chooseBestFeatureToSplit(dataSet)))

    featLabels = []
    myTree = createTree(dataSet, features)
    print(myTree)
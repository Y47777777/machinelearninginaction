'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
import matplotlib.pylab as plt
import numpy as np

def loadDataSet():
    """
    加载数据
    :return:
    """
    # 创建数据集列表和标签列表
    dataMat = []; labelMat = []
    # 打开文件
    fr = open('testSet.txt')
    # 逐行读取
    for line in fr.readlines():
        # 去回车，放入列表
        lineArr = line.strip().split()
        # 添加数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        labelMat.append(int(lineArr[2]))
    # 关闭文件
    fr.close()
    return dataMat, labelMat


def plotDataSet():
    """
    函数说明：绘制数据集
    :return:
    """
    # 加载数据集
    dataMat, labelMat = loadDataSet()
    # 转化成numpy的array
    dataArr = np.array(dataMat)
    # 数据个数
    n = np.shape(dataMat)[0]
    # 正样本
    xcord1 = [];
    ycord1 = []
    # 负样本
    xcord2 = [];
    ycord2 = []
    # 根据数据集标签进行分类
    for i in range(n):
        # 1为正样本
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        # 0为负样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    # 绘制负样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)

    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def sigmoid(inx):
    """
    函数说明：sigmoid函数
    :param inx:
    :return:
    """
    # if inx >= 0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
    #     return 1.0/(1+exp(-inx))
    # else:
    #     return exp(inx)/(1+exp(inx))
    return 1.0/(1+exp(-inx))

def gradAscent(dataMatIn, classLabels):
    """
    梯度上升算法
    :param dataMatIn:
    :param classLabels:
    :return:
    """
    # 转化成mat后转置
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m, n = shape(dataMatrix)
    print(f"m:{m} n:{n}\n")
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    weights_array = []
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        # print(f"h:\n{h}")
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
        weights_array.append(weights)
    return weights, np.array(weights_array)

def plotBestFit(weights):
    """
    绘制数据集
    :param weights:
    :return:
    """
    import matplotlib.pyplot as plt
    # 加载数据集
    dataMat, labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    # 数据分类
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    # 绘制数据点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 绘制回归线
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y.T)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    dataMatrix = np.array(dataMatrix)
    classLabels = np.array(classLabels)
    print(f"dataMatrix:{dataMatrix.shape}\n{dataMatrix}")
    print(f"classLabels:{classLabels.shape}\n{classLabels}")
    # 返回dataMatrix的大小。m为行数,n为列数。
    m, n = shape(dataMatrix)
    # 参数初始化
    weights = ones(n)   #initialize to all ones
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 降低alpha的大小
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            # 随机选择样本
            randIndex = int(random.uniform(0, len(dataIndex)))#go to 0 because of the constant
            # 根据随机选取的一个样本，计算h
            # print(f"randIndex:{randIndex}")
            # print(f"dataMatrix[randIndex]:{dataMatrix[randIndex]}")
            # print(f"weights:{weights}")
            print(f"dataMatrix[randIndex] * weights {j,i}:{dataMatrix[randIndex] * weights}")
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 添加回归系数到数组中
            weights_array = np.append(weights_array, weights, axis=0)
            # 删除已经使用的样本
            del(dataIndex[randIndex])
    # 改变维度
    weights_array = weights_array.reshape(numIter * m, n)
    # 返回
    print(f"weights:{weights.shape}\n{weights}")
    print(f"weights_array:{weights_array.shape}\n{weights_array}")
    return weights, weights_array

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    """
    使用python写的Logistic分类器做预测
    :return:
    """
    # 读取文件
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    # 训练数据和标签的处理
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 训练权重
    trainWeights, weights_array = stocGradAscent1(array(trainingSet), trainingLabels, 200)
    print(f"trainWeights:{trainWeights}")
    errorCount = 0
    numTestVec = 0.0
    # 进行训练结果与实际分类的对比
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 统计错误数据
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

"""
函数说明:绘制回归系数与迭代次数的关系

Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-30
"""
def plotWeights(weights_array1, weights_array2):
    from matplotlib.font_manager import FontProperties
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20,10))
    x1 = np.arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()

if __name__=='__main__':
    # # plotDataSet()
    # dataMat, labelMat = loadDataSet()
    # weights1, weights_array1 = gradAscent(dataMat, labelMat)
    # weights2, weights_array2 = stocGradAscent1(dataMat, labelMat)
    # # plotBestFit(weights)
    # plotWeights(weights_array1, weights_array2)

    colicTest()

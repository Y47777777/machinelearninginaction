'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):
    """
    general function to parse tab -delimited floats
    :param fileName:
    :return:
    """
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    """
    分类数据
    :param dataMatrix: 数据
    :param dimen: 维数
    :param threshVal: 阈值
    :param threshIneq: 小于或大于
    :return:
    """
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    """
    找到数据集上最佳的单层决策树
    :param dataArr:数据矩阵
    :param classLabels:数据标签
    :param D:样本权重
    :return:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    # 数据类型转为mat
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    # 数据大小
    m, n = shape(dataMatrix)
    # 步数，最好的单层决策树， 最好的单层决策树估计
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    #最小误差无穷大
    minError = inf #init error sum, to +infinity
    # 轮询每组数据
    for i in range(n):#loop over all dimensions
        # 取最小值和最大值
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        # 计算步长
        stepSize = (rangeMax-rangeMin)/numSteps
        # 按步长遍历计算从最小到最大，设置为阈值，进行分类，保留最好分类数据
        for j in range(-1, int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                # 阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 预测值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#call stump classify with i, j, lessThan
                # 误差
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 误差最小时记录
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    """
    使用AdaBoost算法训练分类器
    :param dataArr:数据矩阵
    :param classLabels:数据标签
    :param numIt:最大迭代次数
    :return:
        weakClassArr - 训练好的分类器
        aggClassEst - 类别估计累计值
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化权重
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        # 构建单层决策树
        bestStump,error,classEst = buildStump(dataArr, classLabels, D)#build Stump
        #print "D:",D.T
        # 计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        # 存储弱学习算法权重
        bestStump['alpha'] = alpha
        # 储存单层决策树
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        # 计算e的指数项
        expon = multiply(-1*alpha*mat(classLabels).T, classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        # 根据样本权重公式，更新样本权重
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        # 计算类别估计累计值
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        # 计算误差
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate)
        # 误差为0 退出循环
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    """
    roc绘图
    :param predStrengths: 预测强渡
    :param classLabels: 类别标签
    :return:
    """
    import matplotlib.pyplot as plt
    # 绘制光标的位置
    cur = (1.0, 1.0) #cursor
    # 用于计算auc
    ySum = 0.0 #variable to calculate AUC
    # 统计正类的数量
    numPosClas = sum(array(classLabels)==1.0)
    # y轴步长
    yStep = 1/float(numPosClas)
    # x轴步长
    xStep = 1/float(len(classLabels)-numPosClas)
    # 预测强度排序的
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)

if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr, 50)
    print(f"weakClassArr:\n{weakClassArr}")
    plotROC(aggClassEst.T, LabelArr)

'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    """
    加载数据
    :param fileName:
    :param delim:
    :return:
    """
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    """

    :param dataMat:
    :param topNfeat:
    :return:
    """
    # 样本去均值 中心化
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    # 计算样本矩阵的协方差
    covMat = cov(meanRemoved, rowvar=0)
    # 对协方差矩阵进行特征值分解，选取最大的 p 个特征值对应的特征向量组成投影矩阵
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]       #reorganize eig vects largest to smallest
    # 对原始样本矩阵进行投影，得到降维后的新样本矩阵
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean():
    """
    用均值替代nan数据
    :return:
    """
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  #set NaN values to mean
    return datMat

if __name__=='__main__':
    dataMat = mat(loadtxt('testSet.txt'))
    lowMat, reconMat = pca(dataMat, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0],marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0],marker='o', s=50, c='red')
    plt.show()

    # # 加载数据
    # dataMat = replaceNanWithMean()
    # # 去除均值
    # meanVals = mean(dataMat, axis=0)
    # meanRemoved = dataMat - meanVals
    # # 计算协方差
    # covMat = cov(meanRemoved, rowvar=0)
    # # 特征值分析
    # eigVals, eigVects = linalg.eig(mat(covMat))
    # print(eigVals)



import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

"""
函数说明：
    将32*32的二进制图像转化为1*1024的向量
Parameters：
    filename：文件名
Returns：
    returnVect：返回的二进制图像的1024向量
Modify：
    2018-03-12

"""


def img2Vector(filename):
    # 创建1*1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读取一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


"""
函数说明：手写体数字分类

Parameters：
    filename：无
Returns：
    无
Modify：
    2018-03-12

"""


def handwritingClassTest():
    # 测试集的labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1*1024数据存储到traingingMat矩阵中
        trainingMat[i, :] = img2Vector('E:/python/machine learning in action/My Code/chap 02/trainingDigits/%s' % (fileNameStr))
    # 构建KNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型，trainingMat为测试矩阵，hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigirs目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1*1024向量，用于训练
        vectorUnderTest = img2Vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("共错误数据为%d，错误率为%f%%" % (errorCount, errorCount / mTest * 100))


"""
函数说明：main函数

Parameters：
    无
Returns：
    无
Modify：
    2018-03-12

"""
if __name__ == '__main__':
    handwritingClassTest()


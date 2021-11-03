"""
class sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0,
shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)

C：惩罚项，float类型，可选参数，默认为1.0，C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。
相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。

kernel：核函数类型，str类型，默认为’rbf’。可选参数为：
’linear’：线性核函数
‘poly’：多项式核函数
‘rbf’：径像核函数/高斯核
‘sigmod’：sigmod核函数
‘precomputed’：核矩阵
precomputed表示自己提前计算好核函数矩阵，这时候算法内部就不再用核函数去计算核矩阵，而是直接用你给的核矩阵，核矩阵需要为n*n的。

degree：多项式核函数的阶数，int类型，可选参数，默认为3。这个参数只对多项式核函数有用，是指多项式核函数的阶数n，如果给的核函数参数是其他核函数，则会自动忽略该参数。
gamma：核函数系数，float类型，可选参数，默认为auto。只对’rbf’ ,’poly’ ,’sigmod’有效。如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features。
coef0：核函数中的独立项，float类型，可选参数，默认为0.0。只有对’poly’ 和,’sigmod’核函数有用，是指其中的参数c。
probability：是否启用概率估计，bool类型，可选参数，默认为False，这必须在调用fit()之前启用，并且会fit()方法速度变慢。
shrinking：是否采用启发式收缩方式，bool类型，可选参数，默认为True。
tol：svm停止训练的误差精度，float类型，可选参数，默认为1e^-3。
cache_size：内存大小，float类型，可选参数，默认为200。指定训练所需要的内存，以MB为单位，默认为200MB。
class_weight：类别权重，dict类型或str类型，可选参数，默认为None。
给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C。如果给定参数’balance’，则使用y的值自动调整与输入数据中的类频率成反比的权重。
verbose：是否启用详细输出，bool类型，默认为False，此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。
max_iter：最大迭代次数，int类型，默认为-1，表示不限制。
decision_function_shape：决策函数类型，可选参数’ovo’和’ovr’，默认为’ovr’。’ovo’表示one vs one，’ovr’表示one vs rest。
random_state：数据洗牌时的种子值，int类型，可选参数，默认为None。伪随机数发生器的种子,在混洗数据时用于概率估计。

"""

import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC


def img2Vector(filename):
    """
    将32*32的二进制图像转换为1*1024的向量
    :param filename: 文件名
    :return: 返回的二进制图像的1*1024向量
    """
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
    # 返回转换后的1*1024向量
    return returnVect


def handwritingClassTest():
    """
    手写数字分类测试
    :return: 无
    """
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('./digits/trainingDigits')
    print(trainingFileList)
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwlabels中
        hwLabels.append(classNumber)
        # 将每个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2Vector('./digits/trainingDigits/%s' % (fileNameStr))

    clf = SVC(C=200, kernel='rbf')
    clf.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('./digits/testDigits')
    # 错误检测技术
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1*1024向量，用于训练
        vectorUnderTest = img2Vector('./digits/testDigits/%s' % (fileNameStr))
        # 获得预测结果
        classfierResult = clf.predict(vectorUnderTest)
        print("分类返回结果为 %d \t 真实结果为%d " % (classfierResult, classNumber))

        if (classfierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据 \n 错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()

'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
def ecludSim(inA,inB):
    """
    欧氏距离
    :param inA:
    :param inB:
    :return:
    """
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    """
    皮尔逊相关系数
    :param inA:
    :param inB:
    :return:
    """
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA,inB):
    """
    余弦相似度
    :param inA:
    :param inB:
    :return:
    """
    num = float(inA.T * inB)  # 向量inA和向量inB点乘,得cos分子
    denom = la.norm(inA) * la.norm(inB)  # 向量inA,inB各自范式相乘，得cos分母
    return 0.5 + 0.5 * (num / denom)  # 从-1到+1归一化到[0,1]


def standEst(dataMat, user, simMeas, item):
    """
    计算在给定相似度计算方法的前提下，用户对物品的估计评分值
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param simMeas: 相似度度量方法
    :param item: 物品编号
    :return:
    """
    # 数据中行为用于，列为物品，n即为物品数目
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # 用户的第j个物品
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # 寻找两个用户都评级的物品
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
                                      dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], \
                                   dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal
    
def svdEst(dataMat, user, simMeas, item):
    """
    基于SVD的评分估计
    :param dataMat:
    :param user:
    :param simMeas:
    :param item:
    :return:
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # svd分解
    U,Sigma,VT = la.svd(dataMat)
    # 对角阵
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    # 创建新数据矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T,\
                             xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    推荐引擎
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param N: 前N个未评级物品预测评分值
    :param simMeas: 相似度计算
    :param estMethod: 评估函数
    :return:
    """
    # 寻找未评级的物品，nonzeros()[1]返回参数的某些为0的列的编号，dataMat中用户对某个商品的评价为0的列
    # 矩阵名.A：将矩阵转化为array数组类型
    # nonzeros(a)：返回数组a中不为0的元素的下标
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]#find unrated items
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def printMat(inMat, thresh=0.8):
    """
    打印图像矩阵
    :param inMat:
    :param thresh:
    :return:
    """
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print('')

def imgCompress(numSV=3, thresh=0.8):
    """
    实现图像的压缩，允许基于任意给定的奇异值数目来重构图像
    :param numSV:
    :param thresh:
    :return:
    """
    myl = []
    # 打开文本文件，从文件中以数值方式读入字符
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    # 使用SVD对图像进行SVD分解和重构
    U,Sigma,VT = la.svd(myMat)
    # 建立一个全0矩阵
    SigRecon = mat(zeros((numSV, numSV)))
    # 将奇异值填充到对角线
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k, k] = Sigma[k]
    # 重构矩阵
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)

if __name__=='__main__':
    # Data = loadExData()
    # U, Sigma, VT = la.svd(Data)
    # # 由于Sigma是以向量的形式存储，故需要将其转为矩阵，
    # sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # # 也可以用下面的方法，将行向量转化为矩阵，并且将值放在对角线上，取前面三行三列
    # # Sig3=diag(Sigma)[:3,:3]
    # print(Sigma)
    # # 重构原始矩阵
    # print(U[:, :3] * sig3 * VT[:3, :])
    #
    # myMat = mat(loadExData())
    # print(ecludSim(myMat[:, 0], myMat[:, 4]))  # 第一行和第五行利用欧式距离计算相似度
    # print(ecludSim(myMat[:, 0], myMat[:, 0]))  # 第一行和第一行欧式距离计算相似度
    # print(cosSim(myMat[:, 0], myMat[:, 4]))  # 第一行和第五行利用cos距离计算相似度
    # print(cosSim(myMat[:, 0], myMat[:, 0]))  # 第一行和第一行利用cos距离计算相似度
    # print(pearsSim(myMat[:, 0], myMat[:, 4]))  # 第一行和第五行利用皮尔逊距离计算相似度
    # print(pearsSim(myMat[:, 0], myMat[:, 0]))  # 第一行和第一行利用皮尔逊距离计算相似度

    # myMat = mat(loadExData())
    # myMat[0, 3] = myMat[0, 4] = myMat[1, 4] = myMat[2, 3] = 4
    # myMat[4, 1] = 2
    # print(myMat)
    # print(recommend(myMat, 4))
    # print(recommend(myMat, 4, simMeas=ecludSim))
    # print(recommend(myMat, 4, simMeas=pearsSim))

    myMat = mat(loadExData2())
    print(myMat)
    print(recommend(myMat, 1, estMethod=svdEst))
    print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

    # imgCompress(2)



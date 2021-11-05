'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    """
    加载数据集
    :param fileName:
    :return:
    """
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # map函数的主要功能是
        # 对一个序列对象中的每一个元素应用被传入的函数，并且返回一个包含了所有函数调用结果的一个列表
        # 这里从文件中读取的信息是字符型，通过map函数将其强制转化为浮点型
        fltLine = map(float, curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    """
    计算两个向量间的距离
    :param vecA:
    :param vecB:
    :return:
    """
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    """
    随机构建初始质心
    :param dataSet:
    :param k: 数据集
    :return: k个随机质心
    """
    # 求列数
    n = shape(dataSet)[1]
    # 初始化质心
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        # 求每一维数据的最大值和最小值，保证随机选取的质心在边界内
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

def euclDistance(vector1, vector2):
    """
    计算欧式距离
    :param vector1:
    :param vector2:
    :return:
    """
    return sqrt(sum(power(vector2 - vector1, 2)))

# init centroids with random samples
def initCentroids(dataSet, k):
    """
    初始化质心
    :param dataSet: 数据集
    :param k: k值
    :return:
    """
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    kmeans算法
    :param dataSet: 数据
    :param k: k值
    :param distMeas:
    :param createCent:
    :return:
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    # 选取质心
    centroids = createCent(dataSet, k)
    clusterChanged = True
    # 如果聚类循环一直在改变
    while clusterChanged:
        clusterChanged = False
        # 遍历每个数据
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            # 对每个质心求距离
            for j in range(k):
                # 求取数据点到质心的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                # 记录最近的距离和质心id
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            # 记录下质心id和距离的平方
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        # 对每个聚类计算均值得到质心
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]#get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    """
    二分K-均值算法
    该算法首先将所有的点当成一个簇，然后将该簇一分为二。之后选择其中一个簇进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。
    不断重复该过程，直到达到用户所需要的簇个数为止。
    :param dataSet: 数据
    :param k: k个值
    :param distMeas: 求距离的函数
    :return:
    """
    # 数据组数
    m = shape(dataSet)[0]
    # 每个点的聚类信息： 1：属于哪个簇  2：距离的平方
    clusterAssment = mat(zeros((m, 2)))

    # step 1: the init cluster is the whole data set
    # 把所有点都当成一个簇
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 质心list
    centList = [centroid0] #create a list with one centroid
    for j in range(m): #calc initial Error
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2

    # 开始分割聚类
    while (len(centList) < k):
        lowestSSE = inf
        # for each cluster
        for i in range(len(centList)):
            # step 2: 获取第i个聚类的所有点
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]#get the data points currently in cluster i

            # step 3: 将这个簇划分W为两个
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)

            # step 4: 计算分割前后的sse
            sseSplit = sum(splitClustAss[:, 1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)

            # step 5: find the best split cluster which has the min sum of square error
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # step 6: 添加新的聚类，改变其index
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))

        # step 7: 将两个子簇添加到centlist
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])

        # step 8: update the index and error of the samples whose cluster have been changed
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

def showCluster(dataSet, k, centroids, clusterAssment):
    """
    show your cluster only available with 2-D data
    :param dataSet:
    :param k:
    :param centroids:
    :param clusterAssment:
    :return:
    """
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()

if __name__=='__main__':
    ## step 1: load data
    print("step 1: load data...")
    dataSet = []
    fileIn = open('testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
    # dataSet = loadDataSet('testSet.txt')

    ## step 2: clustering...
    print( "step 2: clustering...")
    dataSet = mat(dataSet)
    k = 4
    # centroids, clusterAssment = kMeans(dataSet, k)
    centroids, clusterAssment = biKmeans(dataSet, k)

    ## step 3: show the result
    print("step 3: show the result...")
    showCluster(dataSet, k, centroids, clusterAssment)
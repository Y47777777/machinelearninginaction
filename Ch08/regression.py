'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    """
    加载数据
    :param fileName:
    :return:
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def plotDataSet():
    """
    绘制数据集
    :return:
    """
    xArr,yArr=loadDataSet('ex0.txt')
    #数据个数
    n=len(xArr)
    #样本点
    xcord=[]
    ycord=[]
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #绘制样本点
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def standRegres(xArr,yArr):
    """
    计算回归系数w
    :param xArr: x数据集
    :param yArr: y数据集
    :return:
    """
    # 转为mat
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    # print(f"xMat:\n{xMat}")
    # print(f"yMat:\n{yMat}")
    # print(f"xTx:\n{xTx}")
    # 如果行列式为0 则为奇异矩阵 不能求逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 回归系数 .I为求逆
    ws = xTx.I * (xMat.T*yMat)
    # print(f"ws:\n{ws}")
    return ws

def plotRegression():
    """
    绘制回归曲线和数据点
    :return:
    """
    #加载数据集
    xArr,yArr=loadDataSet('ex0.txt')
    #计算回归系数
    ws=standRegres(xArr,yArr)
    #创建矩阵
    xMat=mat(xArr)
    yMat=mat(yArr)
    #深拷贝
    xCopy=xMat.copy()
    #排序
    xCopy.sort(0)
    #计算对应的y值
    yHat=xCopy*ws
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #绘制回归曲线
    ax.plot(xCopy[:,1],yHat,c='red')
    #绘制样本点
    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def lwlr(testPoint,xArr,yArr,k=1.0):
    """
    使用局部加权线性回归计算回归系数
    :param testPoint: 测试样本点
    :param xArr: x数据集
    :param yArr: y数据集
    :param k: 高斯核的k,自定义参数
    :return:
    """
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建权重对角阵
    weights = mat(eye((m)))
    # 遍历数据集
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]     #
        weights[j, j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    """
    局部加权线性回归测试
    :param testArr:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    # 遍历对每个样本点进行预测
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    # print(f"yHat:\n{yHat}")
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat, xCopy

def plotlwlrRegression():
    """
    绘制多条局部加权回归曲线
    :return:
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 加载数据集
    xArr, yArr = loadDataSet('ex0.txt')
    # 根据局部加权线性回归计算yHat
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)
    # 根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
    # 根据局部加权线性回归计算yHat
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
    # 创建xMat矩阵
    xMat = mat(xArr)
    # 创建yMat矩阵
    yMat = mat(yArr)
    # 排序，返回索引值
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig, axs = plt.subplots(nrows=3, ncols=1,sharex=False, sharey=False, figsize=(10,8))

    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c = 'red')						#绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c = 'red')						#绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c = 'red')						#绘制回归曲线
    axs[0].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点
    axs[1].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点
    axs[2].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点

    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0',FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01',FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003',FontProperties=font)

    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')

    plt.xlabel('X')
    plt.show()

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    """
    误差大小评价函数
    :param yArr:
    :param yHatArr:
    :return:
    """
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):
    """
    岭回归
    :param xMat: x数据集
    :param yMat: y数据集
    :param lam: 缩减系数
    :return:
    ws 回归系数
    """
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    # 行列式等于0 奇异矩阵 不存在逆矩阵
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
    
def ridgeTest(xArr,yArr):
    """
    岭回归测试
    :param xArr: x数据集
    :param yArr: y数据集
    :return:
    """
    xMat = mat(xArr); yMat=mat(yArr).T
    # 求行均值
    yMean = mean(yMat, 0)
    # 数据减去均值
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    # regularize X's 求均值
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    # 行操作 求方差
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    # 数据减去均值除以方差实现标准化
    xMat = (xMat - xMeans)/xVar
    # 30个不同的lambda测试
    numTestPts = 30
    # 初始回归系数矩阵
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        # lambda以e的指数变化，最初是一个非常小的数
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :]=ws.T
    return wMat

def plotwMat():
    """
    绘制回归系数矩阵
    :return:
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    abX, abY = loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    print(f"redgeWeights:\n{redgeWeights}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()


def regularize(xMat):#regularize by columns
    """
    数据标准化
    :param xMat: x数据集
    :return:
    """
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)   #calc mean then subtract it off
    inVar = var(inMat, 0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# def regularize(xMat, yMat):
#     """
#     函数说明:数据标准化
#     Parameters:
#         xMat - x数据集
#         yMat - y数据集
#     Returns:
#         inxMat - 标准化后的x数据集
#         inyMat - 标准化后的y数据集
#
#     """
#     inxMat = xMat.copy()  # 数据拷贝
#     inyMat = yMat.copy()
#     yMean = np.mean(yMat, 0)  # 行与行操作，求均值
#     inyMat = yMat - yMean  # 数据减去均值
#     inMeans = np.mean(inxMat, 0)  # 行与行操作，求均值
#     inVar = np.var(inxMat, 0)  # 行与行操作，求方差
#     inxMat = (inxMat - inMeans) / inVar  # 数据减去均值除以方差实现标准化
#     return inxMat, inyMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    """
    前向逐步线性回归
    :param xArr: x输入数据
    :param yArr: y预测数据
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
    """
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

def plotstageWiseMat():
    """
    函数说明:绘制岭回归系数矩阵

    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    xArr, yArr = loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()
    
from time import sleep
import json
import urllib3
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib3.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
    
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))

if __name__=='__main__':
    # plotDataSet()
    # plotRegression()
    # plotlwlrRegression()

    # abX, abY = loadDataSet('abalone.txt')
    # print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    # yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # print('k=0.1时,误差大小为:', rssError(abY[0:99], yHat01.T))
    # print('k=1  时,误差大小为:', rssError(abY[0:99], yHat1.T))
    # print('k=10 时,误差大小为:', rssError(abY[0:99], yHat10.T))
    # print('')
    # print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    # print('k=0.1时,误差大小为:', rssError(abY[100:199], yHat01.T))
    # print('k=1  时,误差大小为:', rssError(abY[100:199], yHat1.T))
    # print('k=10 时,误差大小为:', rssError(abY[100:199], yHat10.T))
    # print('')
    # print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    # print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    # ws = standRegres(abX[0:99], abY[0:99])
    # yHat = mat(abX[100:199]) * ws
    # print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))

    # 岭回归
    # plotwMat()
    # 逐步前向
    plotstageWiseMat()
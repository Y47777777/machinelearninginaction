"""
class sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True,
    max_iter=None, tol=0.001, solver=’auto’, random_state=None)
alpha：正则化系数，float类型，默认为1.0。正则化改善了问题的条件并减少了估计的方差。较大的值指定较强的正则化。
fit_intercept：是否需要截距，bool类型，默认为True。也就是是否求解b。
normalize：是否先进行归一化，bool类型，默认为False。如果为真，则回归X将在回归之前被归一化。 当fit_intercept设置为False时，将忽略此参数。 当回归量归一化时，注意到这使得超参数学习更加鲁棒，并且几乎不依赖于样本的数量。 相同的属性对标准化数据无效。然而，如果你想标准化，请在调用normalize = False训练估计器之前，使用preprocessing.StandardScaler处理数据。
copy_X：是否复制X数组，bool类型，默认为True，如果为True，将复制X数组; 否则，它覆盖原数组X。
max_iter：最大的迭代次数，int类型，默认为None，最大的迭代次数，对于sparse_cg和lsqr而言，默认次数取决于scipy.sparse.linalg，对于sag而言，则默认为1000次。
tol：精度，float类型，默认为0.001。就是解的精度。
solver：求解方法，str类型，默认为auto。可选参数为：auto、svd、cholesky、lsqr、sparse_cg、sag。
auto根据数据类型自动选择求解器。
svd使用X的奇异值分解来计算Ridge系数。对于奇异矩阵比cholesky更稳定。
cholesky使用标准的scipy.linalg.solve函数来获得闭合形式的解。
sparse_cg使用在scipy.sparse.linalg.cg中找到的共轭梯度求解器。作为迭代算法，这个求解器比大规模数据（设置tol和max_iter的可能性）的cholesky更合适。
lsqr使用专用的正则化最小二乘常数scipy.sparse.linalg.lsqr。它是最快的，但可能在旧的scipy版本不可用。它是使用迭代过程。
sag使用随机平均梯度下降。它也使用迭代过程，并且当n_samples和n_feature都很大时，通常比其他求解器更快。注意，sag快速收敛仅在具有近似相同尺度的特征上被保证。您可以使用sklearn.preprocessing的缩放器预处理数据。
random_state：sag的伪随机种子。
以上就是所有的初始化参数，当然，初始化后还可以通过set_params方法重新进行设定。
"""
import numpy as np
from bs4 import BeautifulSoup
import random

def scrapePage(retX,retY,inFile,yr,numPce,origPrc):
    """
    从页面读取数据，生成retX,retY列表
    :param retX: 数据X
    :param retY: 数据Y
    :param inFile: HTML文件
    :param yr: 年份
    :param numPce:乐高部件数目
    :param origPrc: 原价

    :return: 无
    """
    with open(inFile,encoding='utf-8') as f:
        html=f.read()
    soup=BeautifulSoup(html)
    i=1
    #根据HTML页面结构进行解析
    currentRow=soup.find_all('table',r="%d" % i)
    while(len(currentRow) !=0 ):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)

def setDataCollect(retX, retY):
    """
    函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
    Parameters:
        无
    Returns:
        无

    """
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)  # 2009年的乐高10196,部件数目3263,原价249.99

def usesklearn():
    """
    函数说明:使用sklearn
    Parameters:
        无
    Returns:

    """
    from sklearn import linear_model
    reg = linear_model.Ridge(alpha=.5)
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    reg.fit(lgX, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (
    reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3]))

if __name__ == '__main__':
    usesklearn()


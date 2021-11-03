from sklearn.linear_model import LogisticRegression
"""
优点：实现简单，易于理解，计算代价不高，速度快，存储资源低
缺点：容易欠拟合，分类精度不高，一般只能处理两分类问题，必须借助softmax才能实现多分类问题，且前提是必须线性可分。
"""
"""
函数说明：使用sklearn构建logistics分类器

"""
def colicSklearn():
    frTrain=open('horseColicTraining.txt')
    frTest=open(('horseColicTest.txt'))
    trainingSet=[]
    trainingLabels=[]
    testSet=[]
    testLabels=[]

    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))

    for line in frTest.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))

    classifier=LogisticRegression(solver='liblinear', max_iter=1000).fit(trainingSet, trainingLabels)
    test_accuracy=classifier.score(testSet,testLabels)*100
    print('正确率：%f%%' % test_accuracy)

if __name__=='__main__':
    colicSklearn()


"""
在scikit-learn中，一共有3个朴素贝叶斯的分类算法类。分别是GaussianNB，MultinomialNB和BernoulliNB。
GaussianNB就是先验为高斯分布的朴素贝叶斯；
MultinomialNB就是先验为多项式分布的朴素贝叶斯；
BernoulliNB就是先验为伯努利分布的朴素贝叶斯。

对于新闻的分类，属于多分类问题，可以使用MultinomialNB来完成，假设特征的先验概率为多项式分布：
class sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

alpha：浮点型可选参数，默认为1.0，其实就是添加拉普拉斯平滑，即为上述公式中的λ ，如果这个参数设置为0，就是不添加平滑；
fit_prior：布尔型可选参数，默认为True。布尔参数fit_prior表示是否要考虑先验概率，如果是false,则所有的样本类别输出都有相同的类别先验概率。否则可以自己用第三个参数class_prior输入先验概率，或者不输入第三个参数class_prior让MultinomialNB自己从训练集样本来计算先验概率，此时的先验概率为P(Y=Ck)=mk/m。其中m为训练集样本总数量，mk为输出为第k类别的训练集样本数。
class_prior：可选参数，默认为None。

拟合：
fit：一般的拟合
partial_fit：一般用在训练集数据量非常大，一次不能全部载入内存的时候，这个时候可以把训练集分成若干等分，重复调用该方法来一步步学习训练集。

预测：
predict：常用的预测方法，直接给出测试集的预测类别输出
predict_log_proba：预测出的各个类别对数概率里的最大值对应的类别，也就是predict方法得到类别
predict_proba：它会给出测试集样本在各个类别上预测的概率，预测出的各个类别概率里的最大值对应的类别，也就是predict方法得到类别。
确定要去掉的前deleteN个高频词的个数与最终检测准确率的关系，确定deleteN的取值：
"""


"""
函数说明：切分中文语句
"""
import os
import jieba
import random
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def TextProcessing(folder_path,test_size = 0.2):
    #查看folder_path下的文件
    folder_list=os.listdir(folder_path)
    #训练集
    data_list=[]
    class_list=[]

    #遍历每个子文件夹
    for folder in folder_list:
        #根据子文件夹，生成新的路径
        new_folder_path=os.path.join(folder_path,folder)
        #存放子文件夹下的txt文件的列表
        files=os.listdir(new_folder_path)

        j=1
        #遍历每个txt文件
        for file in files:
            #每类txt样本数最多100个
            if j>100:
                break
            #打开txt文件
            with open(os.path.join(new_folder_path,file),'r',encoding='utf-8') as f:
                raw=f.read()
            #精简模式，返回一个可迭代的generator
            word_cut=jieba.cut(raw,cut_all=False)
            #generator转换为list
            word_list=list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)
            j+=1
        #zip压缩合并，将数据与标签对应压缩
        data_class_list=list(zip(data_list,class_list))
        #将data_class_list乱序
        random.shuffle(data_class_list)
        #训练集与测试集切分的索引值
        index=int(len(data_class_list)*test_size)+1
        #训练集
        train_list=data_class_list[index:]
        #测试集
        test_list=data_class_list[:index]
        #训练集解压缩
        train_data_list,train_class_list=zip(*train_list)
        #测试集解压缩
        test_data_list,test_class_list=zip(*test_list)
        #统计训练集词频
        all_words_dict={}
        for word_list in train_data_list:
            for word in word_list:
                if word in all_words_dict.keys():
                    all_words_dict[word]+=1
                else:
                    all_words_dict[word]=1

        #根据键值倒序排列
        all_words_tuple_list=sorted(all_words_dict.items(),key=lambda
            f:f[1],reverse=True)
        #解压缩
        all_words_list,all_words_nums=zip(*all_words_tuple_list)
        #转换成列表
        all_words_list=list(all_words_list)
        return all_words_list,train_data_list,test_data_list,train_class_list,\
               test_class_list

"""
函数说明：读取文件中的内容并去重

Parameters：
    words_file：文件路径
Returns：
    word_set：读取内容的set集合
Modify：
    2018-03-15

"""
def MakeWordSet(words_file):
    #创建set集合
    words_set=set()
    #打开文件
    with open(words_file,'r',encoding='utf-8') as f:
        #一行一行读取
        for line in f.readlines():
            #去回车
            word=line.strip()
            #有文本，则添加到word_set中
            if len(word)>0:
                words_set.add(word)
    #返回处理结果
    return words_set

def TextFeatures(train_data_list, test_data_list, feature_words):
    # 出现在特征集中，则置1
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    # 返回结果
    return train_feature_list, test_feature_list


"""
函数说明：文本特征提取
Parameters:
    all_words_list - 训练集所有文本列表
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns:
    feature_words - 特征集
Modify:
    2018-03-15
"""
def words_dict(all_words_list,deleteN,stopWords_set=set()):
    #特征列表
    feature_words=[]
    n=1
    for t in range(deleteN,len(all_words_list),1):
        #feature_words额维度为1000
        if n>1000:
            break
        #如果这个词不是数字，且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopWords_set \
                and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
        n+=1
    return feature_words

"""
函数说明：新闻分类器

parameters：
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_class_list - 训练集分类标签
    test_class_list - 测试集分类标签
Returns:
    test_accuracy - 分类器精度
Modify:
    2018-03-15
"""
def TextClassifier(train_feature_list,test_feature_list,train_class_list,test_class_list):
    classifier=MultinomialNB().fit(train_feature_list,train_class_list)
    test_accuracy=classifier.score(test_feature_list,test_class_list)
    return test_accuracy

if __name__=='__main__':
    #文本预处理，训练集存放的地址
    folder_path='E:\python\machine learning in action\My Code\chap 04\SogouC\Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, \
         test_class_list=TextProcessing(folder_path,test_size=0.2)

    #生成stopwords_set
    stopwords_file='stopwords_cn.txt'
    stopwords_set=MakeWordSet(stopwords_file)

    test_accuracy_list=[]
    deleteNs=range(0,1000,20)
    for deleteN in deleteNs:
        feature_words=words_dict(all_words_list,deleteN,stopwords_set)
        train_feature_list,test_feature_list=TextFeatures(train_data_list,
                                                        test_data_list,feature_words)
        test_accuracy=TextClassifier(train_feature_list,test_feature_list,
                                     train_class_list,test_class_list)
        test_accuracy_list.append(test_accuracy)


    plt.figure()
    plt.plot(deleteNs,test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()
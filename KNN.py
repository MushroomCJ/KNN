##2018.11
##by Mushroom

import imp
from sklearn.preprocessing import MinMaxScaler
import scipy
from numpy import *
import numpy as np
import operator
import Dist_func
import GA

#获取训练用数据
def Load_Train_Data():
    #group = array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    #labels = ['A','A','B','B']
    group = np.loadtxt(open("./low_Dim_Data_train.csv"),delimiter=",",skiprows=0)
    labels = np.loadtxt(open("./_names_train.csv"),delimiter=",",skiprows=0)

    #将扩大后的少数集与多数集合并形成新的训练集，lable集也需要重新生成
    Neg_Data,Neg_Size_Ori = GA.Broaden_Neg_Data(3)    #变为n+1倍
    Neg_Size = Neg_Data.shape[0]
    Neg_Lable = [1.0 for i in range(Neg_Size)]

    j=0
    Pos_Data = np.zeros(((labels.shape[0]-Neg_Size_Ori),3))
    for i in range(labels.shape[0]):
        if  labels[i]==0.0:
            Pos_Data[j] = group[i]
            j=j+1
            print("j:",j)
    Pos_Size = Pos_Data.shape[0]
    Pos_Lable = [0.0 for i in range(Pos_Size)]
    
    print("Pos_Data:",Pos_Data)
    print("Neg_Data:",Neg_Data)
    Newgroup = np.concatenate((Pos_Data,Neg_Data),axis=0)
    Newlabels = np.concatenate((Pos_Lable,Neg_Lable),axis=0)
    
    return Newgroup,Newlabels

#KNN分类函数
def classify(dataIn,dataSet,lable,k):
    '''
    dataSize = dataSet.shape[0]
    ##计算欧式距离
    diff = tile(dataIn,(dataSize,1)) - dataSet  #输入与标准值的各维度方向上相减
    sqdiff = diff ** 2   #每一个元素求平方
    squareDist = sum(sqdiff,axis = 1)  #每一行元素相加
    dist = squareDist ** 0.5   #每一行元素开方，相当于N个距离（距离数=行数）
    '''
    dist = Dist_func.Euc_Dist(dataIn,dataSet)
    #print(dist)
    #print(type(dist))

    ##对距离进行排序
    sortedDistIndex = argsort(dist)    ##根据距离从小到大排序，返回下标

    classCount={}
    for i in range(k):
        voteLabel = lable[sortedDistIndex[i]]
        #print(voteLabel)
        #print(type(voteLabel))
        #对选取的k个样本分别所属的类别个数进行统计（计A类有几个，B类有几个）
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
        
    #选取出现类别此数最多的类别
    maxCount = 0
    classes=0.0
    for key,value in classCount.items():
        if (value/k > 1/5)&(key == 1.0):
            #maxCount = value
            classes = key
    return classes



            

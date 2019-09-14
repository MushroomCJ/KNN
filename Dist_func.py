##2018.11
##by Mushroom

import imp
from sklearn.preprocessing import MinMaxScaler
import scipy
from numpy import *
import numpy as np
import operator


##计算欧式距离
def Euc_Dist(dataIn,dataSet):
    dataSize = dataSet.shape[0]  #获取行数
    diff = tile(dataIn,(dataSize,1)) - dataSet  #输入（先经过行扩展）与标准值的各维度方向上相减
    sqdiff = diff ** 2   #每一个元素求平方
    squareDist = sum(sqdiff,axis = 1)  #每一行元素相加
    dist = squareDist ** 0.5   #每一行元素开方，相当于N个距离（距离数=行数）

    return dist

##切比雪夫距离
def Che_Dist(dataIn,dataSet):
    dataSize = dataSet.shape[0]
    temp = tile(dataIn,(dataSize,1))
    for i in range(dataSize):
        temp[i] = abs(temp[i]-dataSet[i])
        temp[i] = max(temp[i])
    
    dist = sum(temp,axis = 1)
    return dist

##曼哈顿距离    
def Man_Dist(dataIn,dataSet):
    dataSize = dataSet.shape[0]
    temp = tile(dataIn,(dataSize,1))
    dist = np.sum(abs(temp-dataSet),axis = 1)

    return dist

##夹角余弦
def Cos_Dist(dataIn,dataSet):
    dataSize = dataSet.shape[0]
    temp = tile(dataIn,(dataSize,1))
    '''
    norm_T = (np.linalg.norm(temp,ord=2,axis=1,keepdims=True)*(np.linalg.norm(dataSet,ord=2,axis=1,keepdims=True)))
    mult_T = sum(temp*dataSet,axis = 1)
    dist = mult_T/norm_T
    '''
    for i in range(dataSize):
        temp[i] = np.dot(temp[i],dataSet[i])/(np.linalg.norm(temp[i])*(np.linalg.norm(dataSet[i])))
    dist = sum(temp,axis = 1)
    return dist






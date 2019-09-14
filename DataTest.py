##2018.11
##by Mushroom

import imp
import sys
import KNN
#from numpy import *
import numpy as np
import pandas as pd
import timeit


def main():
    dataSet,labels = KNN.Load_Train_Data()
    print("Newgroup;",dataSet)
    print(dataSet.shape)
    print("Newlabels;",labels)
    print(labels.shape)

    dataIn = np.loadtxt(open("./low_Dim_Data_test.csv"),delimiter=",",skiprows=0)
    val=np.loadtxt(open("./_names_test.csv"),delimiter=",",skiprows=0)
    k = 100
    '''
    dataOut = KNN.classify(dataIn[197],dataSet,labels,k)
    print("测试数据为:",dataIn[197],"分类结果为：",dataOut)
    print("长度:",len(dataIn))
    '''
    bb=0

    #dataOut={}
    wrong=0
    all=len(dataIn)

    wrong1 = 0
    wrong0 = 0
    predict_0=0
    predict_1=0
    original_1=0
    original_0=0
    for i in range(len(dataIn)):
        #print(i)
        dataOut= KNN.classify(dataIn[i],dataSet,labels,k)
        if dataOut==val[i]:
            bb=bb+1
            #print('ok')
        else:
            wrong=wrong+1
            #print('false')
        if val[i]==1.0:
            original_1 = original_1+1
        if val[i]==0.0:
            original_0 = original_0+1    
        if dataOut==1.0:
            predict_1 = predict_1+1
        if dataOut==0.0:
            predict_0 = predict_0+1    
        if (val[i]==1.0) & (dataOut==0.0):
            wrong1=wrong1+1
        if (dataOut==1.0) & (val[i]==0.0):
            wrong0=wrong0+1
    print(np.sum(val==0.0))
    print(np.sum(val==1.0))
    print("准确率:",(all-wrong)/all)              #准确率
    #print("正确为1的误判:",wrong0/all0)     #正确为1预测为0的个数,准确率
    #print("预测为1中的错误:",wrong1/all1)   #召回率
    print("精确率:",(original_1-wrong1)/(original_1-wrong1+wrong0))
    print("召回率:",(original_1-wrong1)/(original_1))
    #return k


#k = test()
#print("k:",k)
#测量运行时间
elapsedtime = timeit.timeit(stmt=main, number=1)
print('Searching Time Elapsed:(S)', elapsedtime)





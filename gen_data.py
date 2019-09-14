##2018.11
##by ZiBoJia

#coding=utf-8
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
import scipy
from numpy import *

'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''
def eigValPct(eigVals,percentage):
    sortArray=sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num

'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
其中的percentage表示取前多少个特征需要达到的方差占比，默认为0.9'''
def pca(dataMat,percentage=0.99):
    meanVals=mean(dataMat,axis=0)  #对每一列求平均值，因为协方差的计算中需要减去均值
    meanRemoved=dataMat-meanVals
#    print(meanRemoved[0])
    covMat=cov(meanRemoved,rowvar=0)  #cov()计算方差
    eigVals,eigVects=linalg.eig(mat(covMat))  #利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    k=eigValPct(eigVals,percentage) #要达到方差的百分比percentage，需要前k个向量
    eigValInd=argsort(eigVals)  #对特征值eigVals从小到大排序
    eigValInd=eigValInd[:-(k+1):-1] #从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    redEigVects=eigVects[:,eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）
    lowDDataMat=meanRemoved*redEigVects #将原始数据投影到主成分上得到新的低维数据lowDDataMat
    reconMat=(lowDDataMat*redEigVects.T)+meanVals   #得到重构数据reconMat
#    print(reconMat[0])
    return lowDDataMat,redEigVects,meanVals



matrix=np.loadtxt(open("./bank-additional-train.csv"),delimiter=",")
train=matrix[1:4,:]
val=matrix[0:1,:]
for i in range(4,len(matrix)):
    if not(i%4==0):
       train=np.append(train,matrix[i:i+1,:],axis=0)
    else:
       val=np.append(val,matrix[i:i+1,:],axis=0)


#my_matrix = np.loadtxt(open("./_train_test.csv"),delimiter=",")
#my_matrix2 = np.loadtxt(open('./_test_test.csv'),delimiter=",")
#print(len(csv_file2))
#print(len(train))
#print(len(val))
#print(len(my_matrix))
#print(len(my_matrix2))
#np.savetxt('./names_test.csv',my_matrix[:,20:21],delimiter=',')
#print(len(my_matrix[:,20:21]))
#np.savetxt('./namess_test.csv',my_matrix2[:,20:21],delimiter=',')
#print(len(my_matrix2[:,20:21]))
my_matrix=train
my_matrix2=val
low_Dim_Data_test,redEigVects,meanVals=pca(my_matrix[:,0:20])
meanRemoved=my_matrix2[:,0:20]-meanVals
low_Dim_Data_test2=meanRemoved*redEigVects
np.savetxt('./train_test.csv',train,delimiter=',')
np.savetxt('./val_test.csv',val,delimiter=',')
np.savetxt('./_names_train.csv',train[:,20:21],delimiter=',')
np.savetxt('./_names_test.csv',val[:,20:21],delimiter=',')
np.savetxt('./low_Dim_Data_train.csv',low_Dim_Data_test,delimiter=',')
np.savetxt('./low_Dim_Data_test.csv',low_Dim_Data_test2,delimiter=',')

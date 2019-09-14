##2018.11
##by Mushroom

import imp
import pandas as pd
from numpy import *
import numpy as np
from scipy.optimize import fsolve, basinhopping
import operator
import random
import timeit

 
##获取少数类集合D
'''
return：返回少数类集合，少数类集合长度，样本边界
'''
def Get_Neg_Data():
    group = np.loadtxt(open("./low_Dim_Data_train.csv"),delimiter=",",skiprows=0)
    labels = np.loadtxt(open("./_names_train.csv"),delimiter=",",skiprows=0)
    dataSize = labels.shape[0]  #获取行数
    Neg_Data = [0.0 for i in range(3157*3)]
    Neg_Data = np.array(Neg_Data)
    Neg_Data.shape = (3157,3)      #形成合适的array数组以存储少数类

    j=0
    for i in range(dataSize):      #选出少数类并存储
       if  labels[i]==1.0:
           Neg_Data[j] = group[i]
           j=j+1

    Neg_Size = Neg_Data.shape[0]
    Max_dist = 0
    for i in range(Neg_Size):
        diff = tile(Neg_Data[i],(Neg_Size,1)) - Neg_Data
        sqdiff = diff ** 2
        squareDist = sum(sqdiff,axis = 1)
        dist = squareDist ** 0.5
        dist_ave = (sum(dist,axis = 0))/(Neg_Size-1)
        if (dist_ave > Max_dist):        #每个个体平均欧式距离最大值，即集合D的边界
            Max_dist = dist_ave


    #print("Neg_Size:",Neg_Size)
    #print("Neg_Data:",Neg_Data)
    #print("Max_dist:",Max_dist)
    return Neg_Data,Neg_Size,Max_dist


##验证新种群中个体的有效性，根据淘汰机制筛去不适应的个体，无效个体由这个样本的一个亲代代替
##淘汰机制，若平均欧式距离小于初始样本最大平均欧式距离，该个体进入新种群样本
'''
Data_Checked：待验证的种群
Neg_Data：少数类原集
Neg_Size：少数类原集长度
Max_dist：边界值
return：最终的新种群
'''
def Fitness_func(Data_Checked,Neg_Data,Neg_Size,Max_dist):
    dataSize = Data_Checked.shape[0]
    for i in range(dataSize):
        diff = tile(Data_Checked[i],(Neg_Size,1)) - Neg_Data
        sqdiff = diff ** 2
        squareDist = sum(sqdiff,axis = 1)
        dist = squareDist ** 0.5
        dist_ave = (sum(dist,axis = 0))/(Neg_Size-1)
        #print(dist_ave)
        if dist_ave>Max_dist:
            print("invalid data:",Data_Checked[i])
            j = random.randint(0, Neg_Size)
            Data_Checked[i] = Neg_Data[j]

    Checked_Data = Data_Checked
    return  Checked_Data


##随机抽取少数类集合D中的‘r’分之一做初始样本种群（轮盘赌算法不适用）
def Sel_Population(Neg_Data,Neg_Size,r):
    Size = int(Neg_Size/r) + 1
    Init_Date = np.zeros((Size,3))      #形成合适的array数组以存储新种群
    
    for i in range(Size):           #从少数类集合D中随机抽取Size个实例
        sel = random.randint(0, Neg_Size)
        Init_Date[i] = Neg_Data[sel]

    Init_Size = Init_Date.shape[0]
    #print("Init_Size:",Init_Size)
    #print(type(Init_Date))
    
    return Init_Date


##种群交叉：一次交叉两亲代任取一属性交换
'''
Parent_Date：亲代数集
Pc：交叉概率
return：返回子代数集
'''
def Crossover(Parent_Date,Pc=0.7):
    #生成子代数组（生成数组的方法比上面的好）
    m, n = Parent_Date.shape  #(行数，列数)
    Crossover_Date = np.zeros((m, n))
    
    #计算需要进行交叉的个数，生成随机索引
    numbers = np.uint8(m * Pc)
    if numbers % 2 !=0:
        numbers +=1
    index = random.sample(range(m), numbers)
    
    #不交叉的实例进行复制
    for i in range(m):
        if not index.__contains__(i):
            Crossover_Date[i,:] = Parent_Date[i,:]
    
    #crossover
    while len(index)>0:
        a = index.pop()
        b = index.pop()
        #随机产生一个交叉点
        crossoverPoint = random.sample(range(1, n),1)  
        crossoverPoint = crossoverPoint[0]  #化list为整型
        #交换该位
        Crossover_Date[a, 0:crossoverPoint] = Parent_Date[a, 0:crossoverPoint]
        Crossover_Date[a, crossoverPoint:] = Parent_Date[b, crossoverPoint:]
        Crossover_Date[b, 0:crossoverPoint] = Parent_Date[b, 0:crossoverPoint]
        Crossover_Date[b, crossoverPoint:] = Parent_Date[a, crossoverPoint:]

    return Crossover_Date


##基因（属性）变异：某个个体的一个属性按一定概率、规律发生突变
'''
Crossover_Date：交叉后得到的种群
Pm：变异概率
return：经变异操作后的新种群
'''
def Mutation(Crossover_Date,Pm=0.01):
    Muta_Data = np.copy(Crossover_Date)
    m, n = Crossover_Date.shape
    #确定需要变异的基因数
    Gene_num = np.uint8(m * n * Pm)
    #随机抽取gene_num个基因作为变异索引
    MutationGeneIndex = random.sample(range(0, m * n), Gene_num)

    delta = 0
    for i in range(1,20):
        delta += 1/2**i

    #根据索引确定变异基因的具体位置
    for gene in MutationGeneIndex:
        #确定变异基因位于哪个个体（行）
        Unit_Index = gene // n
        #确定变异基因位于当前个体哪个基因位（列）
        Gene_Index = gene % n
        #print("(Unit_Index,Gene_Index)",Unit_Index,Gene_Index)
        #print("X:",Muta_Data[Unit_Index, Gene_Index])
        #mutation：实值变异算子
        s = random.random()
        if s >= 1/2:
            Muta_Data[Unit_Index, Gene_Index] = Muta_Data[Unit_Index, Gene_Index] + 0.5*n*delta
        else:
            Muta_Data[Unit_Index, Gene_Index] = Muta_Data[Unit_Index, Gene_Index] - 0.5*n*delta

        #print("Y:",Muta_Data[Unit_Index, Gene_Index])

    return Muta_Data


##扩展少数类集合
'''
multiple：增大的倍数
New_Neg_Data：扩大的少数类集合
'''
def Broaden_Neg_Data(multiple):
    #获取少数类原集和边界范围
    Neg_Data,Neg_Size,Max_dist = Get_Neg_Data()
    #print("Neg_Data:",Neg_Data)
    #print(Neg_Data.shape)
    New_Neg_Data = Neg_Data
    #获取迭代的初始样本种群
    Init_Date = Sel_Population(Neg_Data,Neg_Size,4)

    for i in range(multiple*4):
        #交叉算法更新种群
        Crossover_Date = Crossover(Init_Date)
        #变异算法更新种群
        Muta_Data = Mutation(Crossover_Date)
        #对一次迭代后的新种群做有效性检测，适者保留
        Checked_Data = Fitness_func(Muta_Data,Neg_Data,Neg_Size,Max_dist)
        #新种群与少数类原集合并，得到新的少数类集合
        New_Neg_Data = np.concatenate((New_Neg_Data,Checked_Data),axis=0)

        print("New_Neg_Data:",New_Neg_Data)
        print(New_Neg_Data.shape)

    return New_Neg_Data,Neg_Size



#New_Neg_Data = Broaden_Neg_Data(1)





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy
from math import pi
from sympy import diff

def data_read():
    # 读取凸轮升程数据
    sheet = pd.read_excel(io='up-list.xls')

    # 抽取数据
    data = []
    for v in range(360):
        data.append(list(sheet.ix[v].values))

    return data

#拟合函数
def func(p,x):
    a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16= p
    return a16*x**16 + a15*x**15 + a14*x**14 + a13*x**13 + a12*x**12 + a11*x**11 + a10*x**10 + a9*x**9 + a8*x**8 + a7*x**7 + a6*x**6 + a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0

#误差函数
def error(p,x,y):
    return func(p,x) - y

#中值差商 用于代替导数
def delta_H1(data):
    delat = []
    for i in range(len(data)):
        if i == 0:
            delat.append((data[i] - 0)/2)
        elif i == len(data) - 1:
            delat.append((0 - data[i])/2)
        else:
            delat.append((data[i+1] - data[i-1])/2)

    return delat

#计算升程H的二阶导数
def delta_H2(data,delta1):
    delta2 = []
    rem = copy.copy(delta1)
    pro = data[0]/2
    rem.insert(0, pro)
    back = -data[-1]/2
    rem.append(back)

    for i in range(1, len(rem)-1):
        delta2.append((rem[i+1] - rem[i-1])/2)

    return delta2

if __name__ == '__main__':
    #初始化数据
    R1 = 325
    R2 = 20
    #读取升程
    datasets = data_read()
    #将数据转化为矩阵
    datasets = np.array(datasets)

    #绘制散点图
    plt.scatter(datasets[:, 0],datasets[:, 1])
    plt.show()

    #拟合H（b）
    pfit = np.polyfit(datasets[87:273,0],datasets[87:273,1], 24)
    y_fun = np.poly1d(pfit)

    #绘制拟合曲线图
    plt.plot(datasets[87:273,0], datasets[87:273,1], color='g')
    plt.show()

    #升程X-C离散点
    C = []
    X = []
    delta1 = delta_H1(datasets[87:273,1])
    delta2 = delta_H2(datasets[87:273,1], delta1)
    for i in range(0,len(delta1)):
        #计算角度C
        fai = math.radians(i+87) + math.atan(delta1[i]/(R1 + R2 + datasets[i+87, 1]))
        C.append(math.degrees(fai))
        #计算位移X
        oo = math.sqrt(delta1[i] ** 2 + (R1 + R2 + datasets[i+87, 1]) ** 2) - R1 - R2
        X.append(oo)

    X = np.array(X)
    C = np.array(C)
    #绘制X-C离散点图
    plt.scatter(C,X)
    plt.show()

    #拟合X-C曲线
    pfit = np.polyfit(C,X,24)
    y_fun = np.poly1d(pfit)
    #绘制X-C曲线
    plt.plot(C,y_fun(C),color='g')
    plt.scatter(datasets[:, 0], datasets[:, 1])
    plt.show()

    #生成X误差
    np.random.seed(1234)
    error_h =np.random.normal(0, 0.1, len(X))
    #绘制图像
    plt.plot(datasets[:, 0], datasets[:, 1], color='r')
    plt.plot(datasets[87:273, 0], error_h+X, color='g')
    plt.show()

    #拟合dH
    pfit = np.polyfit(datasets[87:273, 0],error_h,20)
    y_fun = np.poly1d(pfit)
    plt.plot(datasets[87:273, 0] , error_h)
    plt.plot(datasets[87:273, 0], y_fun(datasets[87:273, 0]))
    plt.show()

    #计算补偿量
    X_ = []
    C_ = []
    for i in range(len(delta1)):
        X_.append(math.sqrt(delta1[i] ** 2 + (R1 + R2 + datasets[i+87, 1] - error_h[i]) ** 2) - R1 - R2)
        C_.append(math.degrees(math.radians(i+87) + math.atan(delta1[i]/(R1 + R2 + datasets[i+87, 1] - error_h[i]))))

    X_ = np.array(X_)
    C_ = np.array(C_)

    #拟合补偿后的曲线
    plt.plot(datasets[:, 0], datasets[:, 1], color='r')
    plt.plot(datasets[87:273, 0], error_h + X, color='g')
    plt.plot(C_,X_+error_h)
    plt.show()
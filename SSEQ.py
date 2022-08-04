
'''
    对于MxN的图像,假设灰度矩阵为f,shape为MxN,则DCT变换可表示为
    T = AfB,其中A:MxM,B:NxN.

    在该程序中，矩阵乘法改为使用torch.bmm()来计算分块后的8*8矩阵乘法。之后分别计算每块的频域熵
'''

import numpy as np
import pandas as pd
from math import sqrt,cos
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
torch.set_printoptions(profile="full")


def RGB_to_Grey(rgb_image):
    '''RGB转灰度图像约定俗成的公式:
    G = 0.299 R + 0.587 G + 0.114 B
    '''
    Grey = np.dot(rgb_image[...,:3],[0.299,0.587,0.114])
    return Grey


#产生DCT变换矩阵 flag标记是否转置
def gen_Matrix(N,batch,flag):
    Matrix = np.zeros((N,N),dtype=float)

    for row in range(N):#遍历行
        if row == 0:#第一行
            for col in range(N):
                Matrix[row,col] = 1 / sqrt(N)
        
        else:
            for col in range(N):
                Matrix[row,col] = sqrt(2/N) * (cos( (np.pi * (2*col+1) * row) / (2*N)))
    
    if flag==1:
        Matrix=Matrix.T
    
    A=torch.unsqueeze(torch.tensor(Matrix),0)
    A=A.repeat(batch,1,1)  #升到三维为batch*8*8的形式

    return A




def cut(grey): #将图像边缘切割 保证长宽为 8的倍数
    Size = list(grey.shape)
    Rows,Cols = Size[0],Size[1]

    Rows=int(Rows/8)
    Cols=int(Cols/8)
    Rows=Rows*8
    Cols=Cols*8

    b=grey[0:Rows,0:Cols]
    return b


def reshape(grey): #图像分块,返回batch*8*8的三维矩阵
     Size = list(grey.shape)
     grey=np.array(grey)
     Rows=Size[0]
     Cols=Size[1]
     Rows=int(Rows/8)
     Cols=int(Cols/8)
     num=np.size(grey)
     num=int(num/64)
     shape = (Rows,Cols, 8, 8)

     strides = grey.itemsize * np.array([64*Cols,8,8*Cols,1])
     a1= np.lib.stride_tricks.as_strided(grey,shape=shape,strides=strides)  #numpy中的高效分块操作np.lib.stride_tricks.as_strided

     tensor=torch.tensor(a1)

     tensor=tensor.reshape(Rows*Cols,8,8)


     return tensor





#主函数
def main():
    
    image = mpimg.imread('img.bmp') #读图像
    Grey = RGB_to_Grey(image) #转为灰度值
  
    
    cutGrey=cut(Grey) 
    
    reshapedGrey=reshape(cutGrey)


    Size=reshapedGrey.size()

    batch=Size[0]
    
    A = gen_Matrix(8,batch,0).to(torch.float32)  #得到DCT变化矩阵
    B = gen_Matrix(8,batch,1).to(torch.float32)  #得到DCT变化矩阵
  
    T = torch.bmm(torch.bmm(A,reshapedGrey.to(torch.float32)),B) #DCT相乘 得到8*8*batch矩阵

  
    #print(T)

   #之后i=batch循环该矩阵，先log2再归一，再求局部熵
   
  
    tmp = torch.zeros(8,8, dtype=torch.float32) #tmp为在一个batch中的8*8二维初始矩阵
    LocalEntropy=torch.zeros(batch,dtype=torch.float32) #LocalEntropy为长度为batch的一维初始矩阵


    for i in range(batch): #每个块求频域熵

        tmp=T[i]  #tmp为在一个batch中的8*8二维矩阵

        square=torch.square(tmp)   #将元素平方

        sum=square.sum().item()    #求平方和

        p=torch.div(square,sum)     #相除归一 得8*8矩阵

        p=-p*torch.log2(p)         #  求局部熵

        p[torch.isnan(p)]=0    #将nan转换为0

        LocalEntropy[i]=p.sum().item()



    print(LocalEntropy.size())
    print(LocalEntropy)  

    entropy_skewness=pd.Series(LocalEntropy).skew()  #求偏度
    print("skewness:")
    print(entropy_skewness)


if __name__ == '__main__':
    main()     


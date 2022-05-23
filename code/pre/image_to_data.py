import numpy as np
from PIL import Image
import pandas as pd

def get_data(i):
    #img = Image.open("E:\Experiment_Data\cut2\stone({1}).bmp".format(i),"r")
    # img = Image.open("D:/cut_//real_stone_image ({0}).bmp".format(i), "r")
    img = Image.open("E:\pycharmprojects\PyTorch-SRGAN\data\myshale\shale({0}).png".format(i), "r")
    image_arr = np.array(img)      #将图片以数组的形式读入变量
    return  image_arr
list1=[]
for k in range(0, 80):
    a1=get_data(k)
    for i in range(80):
        for j in range(80):
            list1.append(a1[i,j])
data_cut=pd.DataFrame([list1])
data_cut=data_cut.T
Y=np.array(data_cut)
X=np.zeros((512000,4),dtype=np.int)
# X=pd.DataFrame(X)
for i in range(80):
    for j in range(80):
        for k in range(80):
            X[i*80*80+j*80+k,0] =i+1
            X[i*80*80+j*80+k,1]=j+1
            X[i*80*80+j*80+k,2]=k+1
            X[i*80*80+j*80+k,3]=Y[i*80*80+j*80+k,0]
data=pd.DataFrame(X,columns=['X','Y','Z','facies'])
data.to_csv("E://Experiment_Data//SR_data",index=False)

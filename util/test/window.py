import torch
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom #pip install pydicom
import cv2 #pip install opencv-python

import shutil

f = open('C:/Users/chenxinyi/Desktop/code/util/filePath/img.txt', 'r')
jsonImg = f.read()
data_newimg = json.loads(jsonImg)

f = open('C:/Users/chenxinyi/Desktop/code/util/filePath/mask.txt', 'r')
jsonMask = f.read()
data_newlabel = json.loads(jsonMask)
print("图片有",len(data_newimg),'张')

from PIL import Image
images = [Image.open(i) for i in data_newimg[0:100]]
temp_image_array=[np.array(i) for i in images]
image_array = np.array(temp_image_array)
# image_array = np.array(images[0])

#可视化展示读取的数据
j = 1
for i in range(51,60):
    plt.subplot(3,3,j)
    plt.imshow(image_array[i],cmap='gray')
    plt.axis('off')
    j += 1
plt.show()

#给定windowing自定义函数
def windowing(img, window_width, window_center):
    #img： 需要增强的图片
    #window_width:窗宽
    #window_center:中心
    minWindow = float(window_center)-0.5*float(window_width)
    new_img = (img-minWindow)/float(window_width)
    new_img[new_img<0] = 0
    new_img[new_img>1] = 1
    return (new_img*255).astype('uint8') #把数据整理成标准图像格式

j = 1
img_ct = windowing(image_array,250,0)
for i in range(51,60):
    plt.subplot(3,3,j)
    plt.imshow(img_ct[i],cmap='gray')
    plt.axis('off')
    j += 1
plt.show()

def clahe_qu(imgs):
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img_res = np.zeros_like(imgs)
    for i in range(len(imgs)):
        img_res[i,:,:]=clahe.apply(np.array(imgs[i,:,:],dtype=np.uint8))
    return img_res

img_clahe=clahe_qu(image_array)

j = 1
for i in range(51,60):
    plt.subplot(3,3,j)
    plt.imshow(img_clahe[i],cmap='gray')
    plt.axis('off')
    j += 1
plt.show()


imag_clahe_window = clahe_qu(img_ct)
j = 1
for i in range(51,60):
    plt.subplot(3,3,j)
    plt.imshow(imag_clahe_window[i],cmap='gray')
    plt.axis('off')
    j += 1
plt.show()


plt.hist(image_array.reshape(-1,),bins=50)
plt.show()

plt.hist(img_ct.reshape(-1,),bins=50)
plt.show()

plt.hist(img_clahe.reshape(-1,),bins=50)
plt.show()

plt.hist(imag_clahe_window.reshape(-1,),bins=50)
plt.show()

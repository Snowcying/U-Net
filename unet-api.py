import os
import math
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import getFile

# kaggle_3m='../data/kaggle_3m/'
# # kaggle_3m='D:/wk11111111111111111/00-资料-代码-数据-课件/资料代码/TCGA颅脑MRI语义分割/数据/kaggle_3m/'
# dirs=glob.glob(kaggle_3m+'*')
# # print(dirs)
#
# data_img=[]
# data_label=[]
# for subdir in dirs:
#     dirname=subdir.split('\\')[-1]
#     for filename in os.listdir(subdir):
#         img_path=subdir+'/'+filename
#         if 'mask' in img_path:
#             data_label.append(img_path)
#         else:
#             data_img.append(img_path)
# # print(data_label)
# data_imgx=[]
# for i in range(len(data_label)):
#     img_mask=data_label[i]
#     img=img_mask[:-9]+'.tif'
#     data_imgx.append(img)
# # print(data_imgx)
#
# data_newimg=[]
# data_newlabel=[]
# for i in data_label:
#     value=np.max(cv2.imread(i))
#     img1 = cv2.imread(i)
#     try:
#         if value>0:
#             # print(value)
#             data_newlabel.append(i)
#             i_img=i[:-9]+'.tif'
#             data_newimg.append(i_img)
#     except:
#         pass
# print(data_newimg[:5])
# print(data_newlabel[:5])
# im=data_newimg[20]
# im=Image.open(im)
idx = {
    '0': 0,
    '10': 1,
    '50': 2,
    '130': 3,
}

def pixel_to_id(array):
    array=array.astype(str)
    ix,jx=array.shape
    for i in range(ix):
        for j in range(jx):
            pixel = array[i][j]
            pixelid = idx[pixel]
            array[i][j] = pixelid
    array = array.astype("int32")
    return array


data_newimg, data_newlabel = getFile.getList()

picSize = 480

train_transformer=transforms.Compose([
    transforms.Resize((picSize, picSize)),
    # transforms.Resize((310,310)),
    # transforms.RandomRotation((90,90)),
    transforms.ToTensor(),
])
test_transformer=transforms.Compose([
    transforms.Resize((picSize, picSize)),
    # transforms.Resize((310,310)),
    # transforms.RandomRotation((90,90)),
    transforms.ToTensor()
])


maskTest = data_newlabel[0]
# mastTest2 = cv2.imread(maskTest)
maskTest=Image.open(maskTest)
pic_array = np.array(maskTest)
print(pic_array.shape)
maskTensor2 = torch.from_numpy(pic_array)
print('torch.max(maskTensor2):  ',torch.max(maskTensor2))

maskTensor = train_transformer(maskTest)
print('torch.max(maskTensor):  ',torch.max(maskTensor))


class BrainMRIdataset(Dataset):
    def __init__(self, img, mask, transformer):
        self.img = img
        self.mask = mask
        self.transformer = transformer

    def __getitem__(self, index):
        img = self.img[index]
        mask = self.mask[index]

        img_open = Image.open(img)
        img_tensor = self.transformer(img_open)

        mask_open = Image.open(mask)
        mask_tensor = self.transformer(mask_open)

        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img)


s=2800
# s = 100
train_img=data_newimg[:s]
train_label=data_newlabel[:s]
test_img=data_newimg[s:]
test_label=data_newlabel[s:]

train_data=BrainMRIdataset(train_img,train_label,train_transformer)
test_data=BrainMRIdataset(test_img,test_label,test_transformer)

dl_train=DataLoader(train_data,batch_size=8,shuffle=True)
dl_test=DataLoader(test_data,batch_size=8,shuffle=True)

img,label=next(iter(dl_train))

# label = label[0].numpy()
# gray_narry = np.array([0.299, 0.587, 0.114])
# x = np.dot(label, gray_narry)
# pic = Image.fromarray(x.astype('uint8'))
# pic.show()

print('label.shape:  ',label.shape)
img,label=next(iter(dl_train))

img2= label[0].numpy()

print('img2.shape:  ',img2.shape)

# print('path',data_newlabel[25])
mask = Image.open(data_newlabel[25])
mask_array = np.array(mask)
print(np.unique(mask_array))
# plt.imshow(mask)
# plt.show()

# plt.imshow(img2)
# plt.show()
#
# plt.figure(figsize=(12,8))
# for i,(img,label) in enumerate(zip(img[:4],label[:4])):
#     img=img.permute(1,2,0).numpy()
#     label=label.numpy()
#     plt.subplot(2, 4, i+1)
#     plt.imshow(img,cmap='gray')
#     plt.subplot(2, 4, i+5)
#     plt.imshow(label)
# plt.show()
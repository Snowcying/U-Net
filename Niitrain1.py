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
# import torch.nn.functional as F
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import getFile

idx = {
    '0': 0,
    '10': 1,
    '50': 2,
    '90': 3,
    '130': 4,
}
# picSize = 864
picSize = 480
# class_num = 5
def pixel_to_id(array):
    array = array.astype(str)
    ix, jx = array.shape
    for ii in range(ix):
        for jj in range(jx):
            pixel = array[ii][jj]
            pixelid = idx[pixel]
            array[ii][jj] = pixelid
    array = array.astype("int32")
    return array


data_newimg, data_newlabel = getFile.getList(0)



# img_transformer = transforms.Compose([
#     transforms.Resize((picSize, picSize)),
#     transforms.ToTensor(),
# ])
#
# label_transformer = transforms.Compose([
#     transforms.Resize((picSize, picSize)),
# ])


img_transformer = transforms.Compose([
    transforms.Resize((picSize, picSize)),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.5),
    # transforms.RandomRotation((0, 360)),
    transforms.ToTensor(),
])
label_transformer = transforms.Compose([
    transforms.Resize((picSize, picSize)),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.5),
    # transforms.RandomRotation((0, 360)),
])


# for index,mask in enumerate(data_newlabel):
#
#     mask_open = Image.open(mask)
#     nplab = np.array(mask_open)
#     uniqueAr = np.unique(nplab)
#     print(uniqueAr)
mask_open = Image.open(data_newlabel[20])
nplab = np.array(mask_open)
uniqueAr = np.unique(nplab)
print(uniqueAr)

def rand_crop(image, label, height=300, width=300):
    '''
    data is PIL.Image object
    label is PIL.Image object
    '''
    crop_params = transforms.RandomCrop.get_params(image, (height, width))
    image = F.crop(image, *crop_params)
    label = F.crop(label, *crop_params)

    return image, label


class Liverdataset(Dataset):
    def __init__(self, img, mask, transformer,mask_transformer):
        self.img = img
        self.mask = mask
        self.transformer = transformer
        self.mask_transformer = mask_transformer

    def __getitem__(self, index):
        img = self.img[index]
        mask = self.mask[index]

        img_open = Image.open(img)
        mask_open = Image.open(mask)

        # img_open,mask_open = rand_crop(img_open,mask_open)

        img_tensor = self.transformer(img_open)
        mask_resize = self.mask_transformer(mask_open)
        mask_Rarray = np.array(mask_resize)
        mask_tensor = torch.from_numpy(mask_Rarray)

        # mask_array = np.array(mask_open)
        # mask_pixleID = pixel_to_id(mask_array)
        # mask_pic = Image.fromarray(mask_pixleID)
        # mask_resize = self.mask_transformer(mask_pic)
        # mask_Rarray = np.array(mask_resize)
        # mask_tensor = torch.from_numpy(mask_Rarray)



        # mask_array = np.array(mask_open)
        # mask_pixleID = pixel_to_id(mask_array)
        # mask_pic = Image.fromarray(mask_pixleID)
        # mask_resize = self.mask_transformer(mask_pic)
        # mask_Rarray = np.array(mask_resize)
        # mask_tensor = torch.from_numpy(mask_Rarray)


        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img)


s = int(len(data_newimg)*0.7)
# s = 100
train_img = data_newimg[:s]
train_label = data_newlabel[:s]
test_img = data_newimg[s:]
test_label = data_newlabel[s:]

train_data = Liverdataset(train_img, train_label, img_transformer, label_transformer)
test_data = Liverdataset(test_img, test_label, img_transformer, label_transformer)

dl_train = DataLoader(train_data, batch_size=8, shuffle=True)
dl_test = DataLoader(test_data,batch_size=8,shuffle=True)

img, label = next(iter(dl_train))

print('label.shape:  ', label.shape)
img, label = next(iter(dl_train))

npLabel =label[0].numpy()
print(np.unique(label[0].numpy()))

plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(img[:4], label[:4])):
    img = img.permute(1, 2, 0).numpy()
    label = label.numpy()
    print("lable:",np.unique(label))
    plt.subplot(2, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 4, i+5)
    plt.imshow(label)
plt.show()

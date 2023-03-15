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
import segmentation_models_pytorch as smp
from tqdm import tqdm
from att_unet import AttentionUnet

idx = {
    '0': 0,
    '10': 1,
    '50': 2,
    '90': 3,
    '130': 4,
}


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

picSize = 480
batch_size = 8
classes_num = 5

img_transformer = transforms.Compose([
    transforms.Resize((picSize, picSize)),
    transforms.ToTensor(),
])
label_transformer = transforms.Compose([
    transforms.Resize((picSize, picSize)),
])


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
        img_tensor = self.transformer(img_open)

        mask_open = Image.open(mask)

        mask_resize = self.mask_transformer(mask_open)
        mask_Rarray = np.array(mask_resize)
        mask_tensor = torch.from_numpy(mask_Rarray)
        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img)


s = int(len(data_newimg)*0.7)
print('train_img=', s,'   test_img=',len(data_newimg)-s)


train_img = data_newimg[:s]
train_label = data_newlabel[:s]
test_img = data_newimg[s:]
test_label = data_newlabel[s:]

train_data = Liverdataset(train_img, train_label, img_transformer, label_transformer)
test_data = Liverdataset(test_img, test_label, img_transformer, label_transformer)

dl_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dl_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# my_model = AttentionUnet(in_channels=1,num_classes=num_classes)
my_model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=classes_num,                      # model output channels (number of classes in your dataset)
)
state_dict = torch.load('./checkpoint/X/6_train_mIou_0.857_test_mIou_0.558.pth')
my_model.load_state_dict(state_dict)
my_model = my_model.to('cuda')


image, mask = next(iter(dl_test))
image = image.to('cuda')
# print(image.shape)
my_model.eval()
pred_mask = my_model(image)

mask = torch.squeeze(mask)
pred_mask = pred_mask.cpu()

num = 3
plt.figure(figsize=(10, 10))
for i in range(num):
    plt.subplot(num, 3, i*num+1)
    plt.imshow(image[i].permute(1,2,0).cpu().numpy())
    plt.subplot(num, 3, i*num+2)
    plt.imshow(mask[i].cpu().numpy())
    plt.subplot(num, 3, i*num+3)
    plt.imshow(torch.argmax(pred_mask[i].permute(1,2,0), axis=-1).detach().numpy())
    plt.savefig('test.png')
plt.show()




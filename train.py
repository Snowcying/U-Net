import os
import math
import random
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

data_newimg,data_newlabel= getFile.getList()

print('len=',len(data_newimg))

train_img_transformer=transforms.Compose([
    transforms.Resize((256,256)),
    # transforms.Resize((480,480)),
    transforms.ToTensor(),
])
train_transformer=transforms.Compose([
    transforms.Resize((256,256)),
    # transforms.Resize((480,480)),
    # transforms.RandomRotation((270, 270)),
    transforms.ToTensor(),
])
test_img_transformer=transforms.Compose([
    transforms.Resize((256,256)),
    # transforms.Resize((480,480)),
    transforms.ToTensor(),
])
test_transformer=transforms.Compose([
    transforms.Resize((256,256)),
    # transforms.Resize((480,480)),
    # transforms.RandomRotation((270, 270)),
    transforms.ToTensor()
])


class BrainMRIdataset(Dataset):
    def __init__(self, img, mask, transformerImg,transformerLabel):
        self.img = img
        self.mask = mask
        self.transformerImg = transformerImg
        self.transformerLabel = transformerLabel

    def __getitem__(self, index):
        img = self.img[index]
        mask = self.mask[index]

        img_open = Image.open(img)
        img_tensor = self.transformerImg(img_open)

        mask_open = Image.open(mask)
        mask_tensor = self.transformerLabel(mask_open)

        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img)


# s=90
# s=1000
s=4200
train_img=data_newimg[:s]
train_label=data_newlabel[:s]
test_img=data_newimg[s:]
test_label=data_newlabel[s:]
# print(test_label)
#
train_data=BrainMRIdataset(train_img,train_label,train_img_transformer,train_transformer)
test_data=BrainMRIdataset(test_img,test_label,train_img_transformer,train_transformer)

dl_train=DataLoader(train_data,batch_size=8,shuffle=True)
dl_test=DataLoader(test_data,batch_size=8,shuffle=True)


import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)

# img,label=next(iter(dl_train))
# pred=model(img)
# print(pred.shape)
model.to('cuda')
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

picSize = 256
# picSize = 480

from tqdm import tqdm
def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    epoch_iou = []

    model.train()
    for x, y in tqdm(testloader):
        x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

            intersection = torch.logical_and(y, y_pred)
            union = torch.logical_or(y, y_pred)
            batch_iou = torch.sum(intersection) / torch.sum(union)
            epoch_iou.append(batch_iou.item())

    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / (total * picSize * picSize)

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    epoch_test_iou = []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(testloader):
            x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

            intersection = torch.logical_and(y, y_pred)
            union = torch.logical_or(y, y_pred)
            batch_iou = torch.sum(intersection) / torch.sum(union)
            epoch_test_iou.append(batch_iou.item())

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / (test_total * picSize * picSize)
    static_dic=model.state_dict()
    res = round(np.mean(epoch_test_iou), 3)
    if res >= 0.85 :
        print('epoch_test_iou=',res)
        torch.save(static_dic,'./checkpoint/{}_train_mIou_{}_test_mIou_{}.pth'.format(epoch,round(np.mean(epoch_iou), 3),round(np.mean(epoch_test_iou), 3)))

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'IOU:', round(np.mean(epoch_iou), 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3),
          'test_iou:', round(np.mean(epoch_test_iou), 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


epochs = 100
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 dl_train,
                                                                 dl_test)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

print(test_acc)


# my_model = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     #encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=2,                      # model output channels (number of classes in your dataset)
# )
# state_dict = torch.load('./checkpoint/25_train_mIou_0.882_test_mIou_0.903.pth')
# my_model.load_state_dict(state_dict)
# my_model=my_model.to('cuda')
#
# image, mask = next(iter(dl_test))
# image = image.to('cuda')
# my_model.eval()
# pred_mask = my_model(image)
#
# mask = torch.squeeze(mask)
#
#
# pred_mask=pred_mask.cpu()
#
# num=3
# plt.figure(figsize=(10, 10))
# for i in range(num):
#     plt.subplot(num, 3, i*num+1)
#     plt.imshow(image[i].permute(1,2,0).cpu().numpy())
#     plt.subplot(num, 3, i*num+2)
#     plt.imshow(mask[i].cpu().numpy())
#     plt.subplot(num, 3, i*num+3)
#     plt.imshow(torch.argmax(pred_mask[i].permute(1,2,0), axis=-1).detach().numpy())
# plt.show()



import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
from torchvision import transforms
import numpy as np

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

dirs = ['../data2/img/','../data2/mask/']
imgs = []
masks = []
for subDir in dirs:
    # print(os.listdir(subDir))
    for fileName in os.listdir(subDir):
        imgPath = subDir + fileName
        if 'mask' in imgPath:
            masks.append(imgPath)
        elif 'img' in imgPath:
            imgs.append(imgPath)
print(len(imgs),len(masks))


dataImg=[]
dataLabel=[]
for index,i in enumerate(masks):
    value = np.max(cv2.imread(i))
    # img1 = cv2.imread(i)
    try:
        if value>0:
            # print(value)
            dataLabel.append(i)
            dataImg.append(imgs[index])
    except:
        pass
print(len(dataImg),len(dataLabel))

img1 = cv2.imread(dataImg[0],0)
print(img1.shape)


train_img_transformer=transforms.Compose([
    # transforms.Resize((256,256)),
    transforms.Resize((480,480)),
    transforms.ToTensor(),
])
train_transformer=transforms.Compose([
    # transforms.Resize((256,256)),
    transforms.Resize((480,480)),
    transforms.RandomRotation((270, 270)),
    transforms.ToTensor(),
])
test_img_transformer=transforms.Compose([
    # transforms.Resize((256,256)),
    transforms.Resize((480,480)),
    transforms.ToTensor(),
])
test_transformer=transforms.Compose([
    # transforms.Resize((256,256)),
    transforms.Resize((480,480)),
    transforms.RandomRotation((270, 270)),
    transforms.ToTensor()
])

# s=90
# train_img=data_newimg[:s]
# train_label=data_newlabel[:s]
# test_img=data_newimg[s:]
# test_label=data_newlabel[s:]


# my_model = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     #encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=2,                      # model output channels (number of classes in your dataset)
# )
# state_dict = torch.load('./checkpoint/25_train_mIou_0.952_test_mIou_0.954.pth')
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
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import getFile

# axle = "x"
# pth = "./checkpoint/2023-9-18-1-13/x/8_train_mIou_0.897_test_mIou_0.706.pth"
#
axle = "y"
pth = "./checkpoint/2023-9-18-1-13/y/16_train_mIou_0.937_test_mIou_0.71.pth"

# axle = "z"
# pth = "./checkpoint/2023-9-18-1-13/z/9_train_mIou_0.922_test_mIou_0.717.pth"

# type = "train"
type = "val"

data_newimg, data_newlabel = getFile.get_file(axle, type)
picSize = 480
batch_size = 4
classes_num = 7

img_transformer = transforms.Compose([
    transforms.Resize((picSize, picSize)),
    transforms.ToTensor(),
])
label_transformer = transforms.Compose([
    transforms.Resize((picSize, picSize)),
])


class Dataset(Dataset):
    def __init__(self, img, mask, transformer, mask_transformer):
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


percent = 0.85
s = int(len(data_newimg) * percent)

s = 0

print('train_img=', s, '   test_img=', len(data_newimg) - s)
train_img = data_newimg[:s]
train_label = data_newlabel[:s]
test_img = data_newimg[s:]
test_label = data_newlabel[s:]

# train_data = Dataset(train_img, train_label, img_transformer, label_transformer)
test_data = Dataset(test_img, test_label, img_transformer, label_transformer)

# dl_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dl_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# my_model = AttentionUnet(in_channels=1,num_classes=num_classes)
my_model = smp.UnetPlusPlus(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=classes_num,  # model output channels (number of classes in your dataset)
)
state_dict = torch.load(pth)
my_model.load_state_dict(state_dict)
my_model = my_model.to('cuda')


def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功！")


def printCountArray(arr):
    # 获取数组中的唯一值和对应的计数
    unique_values, value_counts = np.unique(arr, return_counts=True)
    print("--------------------------------")
    # 打印每个值及其对应的计数
    for value, count in zip(unique_values, value_counts):
        print(f"{value}: {count} 个")
    print("--------------------------------")


def plot():
    rangeCount = 5
    for j in range(rangeCount):
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
            plt.subplot(num, 3, i * num + 1)
            plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())

            plt.subplot(num, 3, i * num + 2)
            plt.imshow(mask[i].cpu().numpy())
            # printCountArray(mask[i].cpu().numpy())

            plt.subplot(num, 3, i * num + 3)
            plt.imshow(torch.argmax(pred_mask[i].permute(1, 2, 0), axis=-1).detach().numpy())
            # printCountArray(torch.argmax(pred_mask[i].permute(1, 2, 0), axis=-1).detach().numpy())

            t = time.time()

            YYMMDD = time.strftime('%Y-%m-%d', time.localtime())
            folder = "./output/" + YYMMDD

            # 检查并创建文件夹
            check_and_create_folder(folder)
            folder = folder + '/' + axle
            check_and_create_folder(folder)

            name = folder + '/' + str(round(t * 1000)) + '.png'
            # if i == num - 1:
            #     plt.savefig(name)
        plt.show()


def getIOU():
    model = my_model
    loss_fn = nn.CrossEntropyLoss()
    # dl_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    epoch_test_iou = []
    #
    # my_iter=iter(dl_test)
    # imgs=[]
    # masks=[]
    # for img,mask in my_iter:
    #     # print(item)
    #     imgs.append(img)
    #     masks.append(mask)
    # print(imgs,masks)

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dl_test):
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

    epoch_test_loss = test_running_loss / len(dl_test.dataset)
    epoch_test_acc = test_correct / (test_total * picSize * picSize)
    res = round(np.mean(epoch_test_iou), 3)
    print("test_iou", res)


if __name__ == '__main__':
    plot()
    # getIOU()

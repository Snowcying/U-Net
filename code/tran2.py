import datetime
import json
import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import getFile

classes_num = 7
picSize = 480
batch_size = 4


# classes_num = 4


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


def timeUtil():
    now = datetime.datetime.now()
    # 格式化输出为字符串
    formatted_datetime = "{0}-{1}-{2}-{3}-{4}".format(now.year, now.month, now.day, now.hour, now.minute)

    # 打印结果
    # print("格式化后的日期和时间：", formatted_datetime)
    return formatted_datetime


timeStr = timeUtil()


def out(list, file):
    jsonImg = json.dumps(list)
    f1 = open(file, 'w')
    f1.write(jsonImg)
    f1.close()


def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功！")


def fit(epoch, model, trainloader, testloader, picSize, saveAxle, loss_fn, optimizer):
    correct = 0
    total = 0
    running_loss = 0
    epoch_iou = []

    model.train()
    for x, y in tqdm(trainloader):
        x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            # print('y_pred=',y_pred)
            # print('y=', y)
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
    static_dic = model.state_dict()
    res = round(np.mean(epoch_test_iou), 3)
    torchSaveDir = './checkpoint/' + timeStr
    check_and_create_folder(torchSaveDir)
    torchSaveDir = torchSaveDir + '/' + saveAxle
    check_and_create_folder(torchSaveDir)
    torch.save(static_dic,
               torchSaveDir + '/{}_train_mIou_{}_test_mIou_{}.pth'.format(epoch,
                                                                          round(np.mean(epoch_iou), 3),
                                                                          round(np.mean(epoch_test_iou), 3)))

    # if res >= 0.85 :
    #     print('epoch_test_iou=',res)
    #     torch.save(static_dic,'./checkpoint/x/{}_train_mIou_{}_test_mIou_{}.pth'.format(epoch,round(np.mean(epoch_iou), 3),round(np.mean(epoch_test_iou), 3)))
    IOU = round(np.mean(epoch_iou), 3)
    test_IOU = round(np.mean(epoch_test_iou), 3)
    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'IOU:', round(np.mean(epoch_iou), 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3),
          'test_iou:', round(np.mean(epoch_test_iou), 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc, IOU, test_IOU


train_loss = ['train_loss']
train_acc = ['train_acc']
test_loss = ['test_loss']
test_acc = ['test_acc']
train_IOU = ['IOU']
test_IOU = ['test_IOU']

date = str(datetime.datetime.now().strftime('%Y-%m-%d-%x')).replace(":", "-")


# filename = date + '_train.txt'
# out(ans, './' + filename)
# out(ans, './checkpoint/' + axle + '/' + filename)


def train(axle, epochs=20,used=False):
    # axle = "y"

    type = "train"
    # type="val"
    # data_newimg, data_newlabel = getFile.get_file(axle, "train")

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=classes_num,  # model output channels (number of classes in your dataset)
    )

    # model = AttentionUnet(in_channels=1,num_classes=5)
    if used:
        pth="./checkpoint/2023-9-18-1-13/z/9_train_mIou_0.922_test_mIou_0.717.pth"
        state_dict = torch.load(pth)
        model.load_state_dict(state_dict)

    model.to('cuda')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_img, train_label = getFile.get_file(axle, "train")
    val_img, val_label = getFile.get_file(axle, "val")

    img_transformer = transforms.Compose([
        transforms.Resize((picSize, picSize)),
        transforms.ToTensor(),
    ])
    label_transformer = transforms.Compose([
        transforms.Resize((picSize, picSize)),
    ])

    # print("train_len:",len(train_img))
    # print("val_len:",len(val_img))

    test_img = val_img
    test_label = val_label

    # percent = 0.85
    # s = int(len(data_newimg) * percent)
    # print('s=', s)
    # train_img = data_newimg[:s]
    # train_label = data_newlabel[:s]
    # test_img = data_newimg[s:]
    # test_label = data_newlabel[s:]
    # train_img, train_label, test_img, test_label = select_random_elements(data_newimg, data_newlabel,0.7)
    print(len(train_img), len(train_label), len(test_img), len(test_label))

    train_data = Dataset(train_img, train_label, img_transformer, label_transformer)
    test_data = Dataset(test_img, test_label, img_transformer, label_transformer)

    dl_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc, epoch_IOU, epoch_test_IOU = fit(epoch,
                                                                                                model,
                                                                                                dl_train,
                                                                                                dl_test, picSize, axle,
                                                                                                loss_fn, optimizer)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
        train_IOU.append(epoch_IOU)
        test_IOU.append(epoch_test_IOU)
        ans = [train_loss, train_acc, test_loss, test_acc, train_IOU, test_IOU]
        filename = timeStr + '_train.txt'
        saveDir = './checkpoint/' + timeStr + '/' + axle + '/'
        out(ans, saveDir + filename)


if __name__ == '__main__':
    # train("x",12,True)
    train("z",10,True)
    train("x")

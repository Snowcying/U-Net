import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import torch.nn.functional as F
from torchvision.transforms import functional as F

import getFile

# picSize = 864
picSize = 480
axle = "x"

type = "train"
# type="val"

img, label = getFile.get_file(axle, type)

img_transformer = transforms.Compose([
    transforms.Resize((picSize, picSize)),
    transforms.ToTensor(),
])
label_transformer = transforms.Compose([
    transforms.Resize((picSize, picSize)),
])

mask_open = Image.open(label[20])
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
        mask_open = Image.open(mask)

        img_tensor = self.transformer(img_open)
        mask_resize = self.mask_transformer(mask_open)
        mask_Rarray = np.array(mask_resize)
        mask_tensor = torch.from_numpy(mask_Rarray)

        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img)


s = int(len(img) * 0.7)
# s = 100
train_img = img[:s]
train_label = label[:s]
test_img = img[s:]
test_label = label[s:]

train_data = Dataset(train_img, train_label, img_transformer, label_transformer)
test_data = Dataset(test_img, test_label, img_transformer, label_transformer)

dl_train = DataLoader(train_data, batch_size=8, shuffle=True)
dl_test = DataLoader(test_data, batch_size=8, shuffle=True)

img, label = next(iter(dl_train))

print('label.shape:  ', label.shape)
img, label = next(iter(dl_train))

npLabel = label[0].numpy()
print(np.unique(label[0].numpy()))


def printCountArray(arr):
    # 获取数组中的唯一值和对应的计数
    unique_values, value_counts = np.unique(arr, return_counts=True)

    # 打印每个值及其对应的计数
    for value, count in zip(unique_values, value_counts):
        print(f"{value}: {count} 个")


plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(img[:4], label[:4])):
    img = img.permute(1, 2, 0).numpy()
    label = label.numpy()
    print("label:", np.unique(label))
    printCountArray(label)
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 4, i + 5)
    plt.imshow(label)
plt.show()

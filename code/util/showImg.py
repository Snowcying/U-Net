import os
import random
import cv2
import numpy as np

import matplotlib.pyplot as plt


def get_random_files(folder_path, n):
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                  os.path.isfile(os.path.join(folder_path, file))]
    random_files = random.sample(file_paths, n)
    return random_files


# dataSource="F:/unet_data/aug_img/mask"
dataSource = "F:/unet_data/0917_14_7_train/x"
# 创建一组图片数据
# images = [np.random.rand(100, 100) for _ in range(9)]  # 这里使用随机生成的图片数据
images = get_random_files(dataSource, 10)

print(images)

# 创建图形和子图
# plt.figure(figsize=(10, 10))

# 在每个子图上显示图片
# num=3
# for i in range(num):
#     im=plt.imread(images[i])
#
#     plt.subplot(num, 3, i * num + 1)
#     plt.imshow(im)
#     plt.subplot(num, 3, i * num + 2)
#     plt.imshow(im)
#     plt.subplot(num, 3, i * num + 3)
#     plt.imshow(im)


fig, axes = plt.subplots(3, 3, figsize=(8, 8))

# 在每个子图上显示图片
for ax, image in zip(axes.flatten(), images):
    value = np.max(cv2.imread(image))
    print("value",value)
    image = plt.imread(image)
    ax.imshow(image, cmap='gray')
    ax.axis('off')

# 调整子图之间的间距
plt.tight_layout()

# 展示图形
plt.show()

# 展示图形
plt.show()

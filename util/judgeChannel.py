import os.path
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# 查看图片与通道数关系
import numpy as np
import torch


def three_to_one():
    img_path = 'C:/Users/Administrator/Desktop/data/data3channel/1013data/'
    save_img_path = 'C:/Users/Administrator/Desktop/data/test/'
    for file in os.listdir(img_path):
        if 'mask' in file:
            image = Image.open(os.path.join(img_path, file))
            image=image.convert("L")
            # print()
            image.save(os.path.join(save_img_path, file))


# three_to_one()

# pathFile = "../../data/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1_mask.tif"
# pathFile = "../../origin/masks/1 (2).jpg"
# pathFile = 'C:/Users/Administrator/Desktop/data/1013data/axial0077_mask.png'
# pathFile = 'D:/wk11111111111111111/00/code/TCGA MRI/code/util/image/1/P00176954_0.png'
pathFile = 'D:/dataImage/finalCropedData/1/axial0001P00047619.png'
# pathFile = 'C:/Users/Administrator/Desktop/data/data3channel/1013data/axial0076_mask.png'
# print(os.path.exists(pathFile))
# img1 = Image.open(pathFile)
# img2 = Image.open("D:/code/testphoto/2.png")
# img3 = Image.open("D:/code/testphoto/3.png")
# print(len(img1.split()))

#
img1 = Image.open(pathFile)

print(len(img1.split()))
# img1 = img1.convert("L")
img_array = np.array(img1)
num0 = 0
num255 =0
num = 0

for item in img_array:
    # for x in item:
        for y in item:
            if y == 0:
                num0+=1
            elif y ==255:
                num255+=1
            else:
                num+=1
print(num0,num255,num)
            # if y>0 and y!=255:
                # print(y)
                # num = num+1
# for item in img_array:
#     for x in item:
#         for y in x:
#             if y>0 and y!=255:
#                 # print(y)
#                 num = num+1
# print(num)
print(img_array.shape)
plt.imshow(img_array)
plt.show()


# img1 = cv2.imread(pathFile)
# print(img1.shape)
# cv2.imshow('input', img1)

# ir_img = cv2.imread(pathFile, cv2.IMREAD_GRAYSCALE)
#         # ir_imgr, ir_imgg, ir_imgb = ir_img.split()
# print(ir_img.shape)
# ir_img = ir_img.astype(np.uint16)*4
# cv2.imwrite('F:/wrong/img_0317.png', ir_img)
# cv2.imshow('img',ir_img)
# print(ir_img.shape)
# cv2.waitKey(0)

# print(img1.shape)
# h,w=img1.shape
# for row in range(h):
#     for col in range(w):
#         value = img1[row][col]
#         if value>10:
#             img1[row,col]=255
#         else:
#             img1[row,col]=0
# cv2.imshow('result', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# imgTensor = torch.from_numpy(img1)
# imgNumpy = torch.tensor(imgTensor)
# print(imgTensor.shape)
# print(imgNumpy)

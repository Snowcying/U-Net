# 筛选出有效图片 并保存到paht2路径中

import cv2
import numpy as np
import os
import shutil

path = r'D:\dataImage\image\P00179632\data'
num = 0


# path1 = "D:/dataImage/image/"
path1 = "D:/dataImage/imageAll/"
path2 = "D:/dataImage/data/"
dirList = os.listdir(path1)


def copyfile(old_file_path,new_folder_path):
    shutil.copy(old_file_path, new_folder_path)

all = 1

for dirName in dirList:
    pathData = path1+dirName+'/data'
    pathMask = path1+dirName+'/mask'
    for file in os.listdir(pathData):
        pic = pathData+'/'+file
        # maskName = file.split('.png')[0] + '_mask' + '.png'
        flag = 'mask' in file
        if flag:
            originName = file.split('_mask')[0]
            originPath = pathData+'/'+originName+'.png'
            # print(os.path.exists(originPath))
            # print(originPath)
            # print(pic)

            # value = np.max(cv2.imread(pic))

            # value = 1

            # if value>0:
            #     copyfile(pic, path2)
            #     copyfile(originPath, path2)
            #     num=num+1

            if all == 1:
                copyfile(pic, path2)
                copyfile(originPath, path2)
        # print(pic)
        #         os.rename(os.path.join(pathData, file), os.path.join(pathMask, file))
        #         os.rename(os.path.join(pathData, file), os.path.join(pathMask, file))
                # num+=1




# for file in os.listdir(path):
#     filePath  = path+'/'+file
#     # print(filePath)
#     value = np.max(cv2.imread(filePath))
#     flag = 'mask' in file

    # if value>0 and flag:
        # print(file)
        # num +=1
print(num)
# print(value)
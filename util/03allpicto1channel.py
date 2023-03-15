# 把mask图片转为单通道

import picTo1Channel
import os
import shutil
from PIL import Image
img_path = 'D:/dataImage/data/'
save_img_path = 'D:/dataImage/finalAllData/'


def copyfile(old_file_path,new_folder_path):
    shutil.copy(old_file_path, new_folder_path)

len = len(os.listdir(img_path))
num = 1

def logNum(num):
    if num%10==0:
        print(str(num)+'/'+str(len))

for file in os.listdir(img_path):
    if 'mask' in file:
        filename = img_path+file
        # print(filename)
        picTo1Channel.trans(filename,save_img_path)
        logNum(num)
        num+=1
    else:
        filename = img_path+file
        copyfile(filename,save_img_path)
        logNum(num)
        num+=1
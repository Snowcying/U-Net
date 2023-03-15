import cv2
import glob
import os
import numpy as np
import json
import random

# kaggle_3m='../data/kaggle_3m/'
# houzhui = '.tif'

# kaggle_3m= 'C:/Users/Administrator/Desktop/data/data/'
# kaggle_3m = 'D:/dataImage/finalCropedData/'
# kaggle_3m= 'D:/wk11111111111111111/00/code/TCGA MRI/code/util/image/'

# kaggle_3m= 'F:/unet_data/gz_out_data_x/'    # x轴
# kaggle_3m= 'F:/unet_data/clahe_data_x/'    #  clahe  x轴
# kaggle_3m= 'D:/1Gdata/data/y/'
kaggle_3m= 'F:/unet_data/aug_img/aug/'

houzhui = '.png'

# kaggle_3m='D:/wk11111111111111111/00-资料-代码-数据-课件/资料代码/TCGA颅脑MRI语义分割/数据/kaggle_3m/'
dirs=glob.glob(kaggle_3m+'*')
# print(dirs)

def getList(isRead=0):
    if isRead == 1:
        print("正在生成文件")
        data_img=[]
        data_label=[]
        for subdir in dirs:
            dirname=subdir.split('\\')[-1]
            for filename in os.listdir(subdir):
                img_path=subdir+'/'+filename
                if 'mask' in img_path:
                    data_label.append(img_path)
                else:
                    data_img.append(img_path)
        # print(data_label)
        data_imgx=[]
        for i in range(len(data_label)):
            img_mask=data_label[i]
            img=img_mask[:-9]+ houzhui
            data_imgx.append(img)
        # print(data_imgx)

        data_newimg=[]
        data_newlabel=[]
        for i in data_label:
            value=np.max(cv2.imread(i))
            try:
                if value == 0:
                #不筛选
                # if value > 0:
                    ranNum = random.random()
                    # if ranNum>0.68:
                    #     data_newlabel.append(i)
                    #     i_img=i[:-9]+houzhui
                    #     data_newimg.append(i_img)
                else:
                    data_newlabel.append(i)
                    i_img=i[:-9]+houzhui
                    data_newimg.append(i_img)
            except:
                pass
        jsonImg = json.dumps(data_newimg)
        f1 = open('util/filePath/img.txt', 'w')
        f1.write(jsonImg)
        f1.close()
        jsonMask = json.dumps(data_newlabel)
        f2 = open('util/filePath/mask.txt', 'w')
        f2.write(jsonMask)
        f2.close()
        return data_newimg,data_newlabel
    else:
        print("正在读取文件")
        f = open('util/filePath/img.txt', 'r')
        jsonImg = f.read()
        data_newimg = json.loads(jsonImg)

        f = open('util/filePath/mask.txt', 'r')
        jsonMask = f.read()
        data_newlabel = json.loads(jsonMask)
        print("图片有",len(data_newimg),'张')
        return data_newimg,data_newlabel
        # print(data_newlabel,data_newimg)
        # print(data_newimg[:5])
        # print(data_newlabel[:5])
    # im=data_newimg[20]
    # im=Image.open(im)


# a1,a2=getList(1)

# print(len(a1),a1[:5])
# print(len(a2),a2[:5])
# a = ['1123','2a','ag','b:fda']
# # print(a)
# a=json.dumps(a)
# f = open('./filePath/img.txt','w')
# f.write(a)
# f.close()
#
# f= open('./filePath/img.txt','r')
# file = f.read()
# file = json.loads(file)
# # print(file)
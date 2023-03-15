import cv2 as cv
import numpy as np
import scipy.misc
import glob
import os
import cv2

def image_pixel(image_path,image2_path,dirname):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    img2 = cv.imread(image2_path, cv.IMREAD_COLOR)
    # cv.imshow('input', img)

    h, w, c = img.shape
    # 遍历像素点，修改图像b,g,r值
    for row in range(h):
        for col in range(w):
            b,g,r = img[row, col]
            # img[row, col] = (255 - b, 255 - g, 255 - r)
            # img[row, col] = (255 - b, g, r)
            # img[row, col] = (255 - b, g, 255 - r)
            x = 10
            # print(b,g,r)
            if(b>x or g>x or r>x):
                # print(b,g,r)
                img[row, col] = (255,255,255)
            else:
                # print(b,g,r)
                img[row,col]=(0,0,0)

    # cv.imshow('result', img)
    # cv.imwrite('images/result.jpg', img)
    savePath = './img/'+dirname
    savePath2 = './img2/'+dirname
    scipy.misc.toimage(img, cmin=0.0, cmax=...).save(savePath)
    scipy.misc.toimage(img2, cmin=0.0, cmax=...).save(savePath2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def imgToOne(image_path,dirname):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(dirname, img)


kaggle_3m='../../origin/'
# kaggle_3m='D:/wk11111111111111111/00-资料-代码-数据-课件/资料代码/TCGA颅脑MRI语义分割/数据/kaggle_3m/'
dirs=glob.glob(kaggle_3m+'*')
# print(dirs)

data_img=[]
data_label=[]
for subdir in dirs:
    dirname=subdir.split('\\')[-1]
    for filename in os.listdir(subdir):
        img_path=subdir+'/'+filename
        if 'mask' in img_path:
            data_label.append(img_path)
        elif 'img' in img_path:
            data_img.append(img_path)

# print(data_img[:5])
# print(data_label[:5])
# print('len1',len(data_img))
# data_imgx=[]

# img = Image.open('../data2/mask/1 (1).jpg')


# value = np.max(cv2.imread(data_label[0]))
# print(value)
data_newimg=[]
data_newlabel=[]

valueDic = {}
for v in range(31):
    v = str(v)
    valueDic[v]=0

for index,i in enumerate(data_label):
    value = np.max(cv2.imread(i))
    img1 = cv2.imread(i)
    try:
        if value>0:
            # print(value)
            # valueStr = str(value)
            # valueDic[valueStr]+=1
            dirname = i.split('/')[-1]
            fileImg = i
            fileMask = data_img[index]
            image_pixel(fileImg,fileMask,dirname)
            # data_newlabel.append(i)
            # data_newimg.append(data_img[index])
    except:
        pass
print(len(data_newlabel))

# image_pixel(data_newlabel[10],'1.jpg')
# print(valueDic)
# print(len(data_newimg))
# print(data_newimg)
# for i in data_newlabel:
#     dirname = i.split('/')[-1]
#     image_pixel(i,dirname)
#     print(dirname)
# image_pixel('../../data2/mask/1 (10).jpg')
# image_pixel('./outfile.jpg')
#
# image_path='../../data2/mask/1 (10).jpg'
# img = cv.imread(image_path, cv.IMREAD_COLOR)
# scipy.misc.toimage(img, cmin=0.0, cmax=...).save('outfile.jpg')
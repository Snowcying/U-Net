import glob
import os

import SimpleITK as sitk
import cv2
import numpy as np
from PIL import Image

pixelList = []

# dir = "D:/1Gdata/mask/"
dir = "F:/unet_data/gz_data/"  # 数据源
# savaPath = 'D:/1Gdata/data/y/1/'
# savaPath = "F:/unet_data/gz_out_data/"
# savaPath = "F:/unet_data/Pelvis_data_x/1/"
savaPath = "F:/unet_data/clahe_data_x/1/"
save_img_path = "F:/unet_data/aug_img/img/"
save_mask_path = "F:/unet_data/aug_img/mask/"

pic_pixel = {
    # 'Bladder':11,
    # 'CTV':22,
    # 'GTV':33,
    # 'Intestines':44,
    # 'MM':55,
    # 'Patient':66,
    # 'Rectum':77,
    # 'Signoid':88
}
pic_dic = {
    'image': 0,
    'mask_Bladder': 0,  # 膀胱
    'mask_CTV': 0,
    'mask_GTV': 0,
    'mask_Intestines': 0,
    'mask_MM': 0,
    'mask_Patient': 0,
    'mask_patient': 0,
    'mask_Rectum': 0,
    'mask_Signoid': 0,
    'mask_GTV-ln': 0,
    'mask_GTV-T': 0,
    'mask_Utethra': 1,  # 尿道
    'mask_Colon': 1,  # 结肠
    'mask_Pelvis': 1,  # 骨盆
    'mask_Bladder': 1,  # 膀胱

}
pic_dic_ac = {}
pic_list_ac = []

num = 0

for key in pic_dic:
    if pic_dic[key] == 1:
        pic_dic_ac[key] = 1
        pic_list_ac.append(key)
        num += 1

for index, value in enumerate(pic_list_ac):
    # pixelGap = 200/(num+1)
    pixelGap = 1

    pic_pixel[value] = int((index + 1) * pixelGap)
print(pic_pixel)
# print(num)


patientList = []


def getdic(dir):
    dic = {}
    dirList = glob.glob(dir + '*')
    keys = pic_dic.keys()
    for subDir in dirList:
        # patientList.append(subDir.split('\\')[-1])
        for file in os.listdir(subDir):
            fileName = file.split('.nii.gz')[0]
            dic[fileName] = 0
            if fileName in keys:
                if pic_dic[fileName] == 1:
                    dic[fileName] = 1
    # keys2 = dic.keys()
    # for key in keys2:
    #     print(key.split('mask_')[-1])
    return dic


def getniiList(dir, dic):
    dirList = glob.glob(dir + '*')
    keys = pic_dic.keys()
    niiList = []
    for subDir in dirList:
        subNiiList = []
        for file in os.listdir(subDir):
            fileName = file.split('.nii.gz')[0]
            # print(fileName)
            if dic[fileName] == 1:
                filePath = subDir + '/' + file
                subNiiList.append(filePath)
        niiList.append(subNiiList)
    return niiList


def showUnlable(niiList):
    for index, nii in enumerate(niiList):
        if len(nii) < num:
            Lable = []
            for x in nii:
                fileX = x.split('/')[-1].split('.nii.gz')[0]
                Lable.append(fileX)
            unLable = set(Lable) ^ set(pic_list_ac)
            print(patientList[index], "缺少", num - len(nii), "个标签", unLable)


# showUnlable(niiList)

# print(patientList)

def getfile1(niiList):  # 一个病人含有全部标签
    list = []
    for nii in niiList:
        if len(nii) == num:
            list.append(nii)
    return list


def getList():
    dic = getdic(dir)  # 得到总表
    print(dic)

    niiList = getniiList(dir, dic)  # 筛选出有所需标签的文件
    print(niiList)
    print(len(niiList))

    list = getfile1(niiList)  # 筛选规则1：一个病人含有全部标签
    print(list)
    print(len(list))
    return list


list = getList()  # 所需mask文件的地址
print(list)


def windowing(img, window_width, window_center):
    # img： 需要增强的图片
    # window_width:窗宽
    # window_center:中心
    minWindow = float(window_center) - 0.5 * float(window_width)
    new_img = (img - minWindow) / float(window_width)
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    return (new_img * 255).astype('uint8')  # 把数据整理成标准图像格式


def merge_one_patient(list1, name):
    npList = []
    for i, file in enumerate(list1):
        fileMask = file.split('/')[-1].split('.nii.gz')[0]  # mask_xxxx
        # print(fileMask)
        pixel = pic_pixel[fileMask]
        # print(pixel)
        itk_img = sitk.ReadImage(file)
        imgs = sitk.GetArrayFromImage(itk_img)
        for index in range(imgs.shape[0]):
            img1 = imgs[index, :, :]
            # img1 = imgs[:, :, index]
            img1[img1 == 255] = pixel
            if len(npList) < imgs.shape[0]:
                npList.append(img1)
            else:
                img2 = npList[index]
                newImg = np.maximum(img1, img2)
                npList[index] = newImg
    # image_array = np.array(npList)
    # img_ct = windowing(image_array, 250, 0)
    for i, img in enumerate(npList):
        fileName = name + '_' + str(i) + '_mask.png'
        # savePathFile = savaPath+fileName
        savePathFile = save_mask_path + fileName

        im = Image.fromarray(img)
        im.save(savePathFile)


def export_mask(list1):
    for i, subList in enumerate(list1):
        # name = patientList[i]
        name = subList[0].split('\\')[-1].split('/')[0]  # P00xxxxx
        merge_one_patient(subList, name)


# mergeOnePatient(list[0],'qwe')


export_mask(list)


def clahe_qu(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_res = np.zeros_like(imgs)
    for i in range(len(imgs)):
        img_res[i, :, :] = clahe.apply(np.array(imgs[i, :, :], dtype=np.uint8))
    return img_res


def exportImage(list1):
    for i, subList in enumerate(list1):
        # name = '1'
        name = subList[0].split('\\')[-1].split('/')[0]
        imagePath = subList[0].split('mask_')[0] + 'image.nii.gz'
        itk_img = sitk.ReadImage(imagePath)
        imgs = sitk.GetArrayFromImage(itk_img)
        npList = []
        for index in range(imgs.shape[0]):
            # img1 = imgs[index,:,:]
            img1 = imgs[index, :, :]
            new_array = (np.maximum(img1, 0) / img1.max()) * 255.0
            new_array = np.uint8(new_array)
            npList.append(new_array)

            im = Image.fromarray(new_array)

            # print(type(im))
            fileName = name + '_' + str(index) + '.png'
            savePathFile = savaPath + fileName
            # print(savePath)
            # im.save(savePathFile)

        # print(imagePath)
        image_array = np.array(npList)
        img_clahe = clahe_qu(image_array)
        for index in range(imgs.shape[0]):
            im = Image.fromarray(img_clahe[index])
            fileName = name + '_' + str(index) + '.png'
            # savePathFile = savaPath+fileName
            savePathFile = save_img_path + fileName
            # print(savePath)
            im.save(savePathFile)


exportImage(list)

# def showNii(img):
#     endNum = 0
#     for i in range(img.shape[0]):
#         img1 = img[i,:,:]
#         # print(np.max(img1)
#         if np.max(img1)>0 and endNum<5:
#             # pixelList.append(np.max(img1))
#             # img1[img1==255]=random.randint(20,255)
#             # img1[img1==0]=25
#
#             new_array = (np.maximum(img1, 0) / img1.max()) * 255.0
#             new_array = np.uint8(new_array)
#             # plt.imshow(new_array, cmap='gray')
#             # plt.show()
#
#             # print(np.max(new_array))
#             im = Image.fromarray(new_array)
#
#
#             fileName = 'pic'+str(endNum)+'.png'
#             savePath = './image/'+fileName
#             # im = im.convert("RGB")
#             im.save(savePath)
#             # print(np.max(img1))
#             # plt.imshow(img[i, :, :], cmap='gray')
#             # plt.show()
#             endNum=endNum+1
#     # print(len(pixelList))
# niiFile = r'D:\1Gdata\mask\P00047619\image.nii.gz'
#
# itk_img = sitk.ReadImage(niiFile)
# img = sitk.GetArrayFromImage(itk_img)
# print(img.shape)  # (155, 240, 240) 表示各个维度的切片数量

# for index in range(img.shape[0]):

# showNii(img)

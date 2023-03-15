from PIL import Image
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
import sys
import random
import SimpleITK as sitk
path1=  "../origin/imgs/1_(50).jpg"
path2 = "../origin/masks/1 (50).jpg"
img = Image.open("../origin/imgs/1_(50).jpg")
# img1 = Image.open("./img(1).jpg")
# img = np.rot90(img, 3)
mask  = Image.open("../origin/masks/1 (50).jpg")
# img.show()
# mask.show()

# img1 = cv2.imread(path1)
# img2 = cv2.imread(path2)
# cv2.imshow('img',img1)
# cv2.imshow('mask',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def InitDicomFile():
#     infometa = pydicom.dataset.Dataset()
#     infometa.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
#     infometa.ImplementationVersionName = 'Python ' + sys.version
#     infometa.MediaStorageSOPClassUID = 'CT Image Storage'
#     infometa.FileMetaInformationVersion = b'\x00\x01'
#     return infometa


dicomFile = "D:\dataImage\dicomSeries\IMG0001.dcm"

pixelList = []

pic_pixel = {
    'Bladder':11,
    'CTV':22,
    'GTV':33,
    'Intestines':44,
    'MM':55,
    'Patient':66,
    'Rectum':77,
    'Signoid':88
}
pic_List ={
    'Bladder': 1,
    'CTV': 0,
    'GTV': 1,
    'Intestines': 1,
    'MM': 0,
    'Patient': 0,
    'Rectum': 0,
    'Signoid': 0
}


def showNii(img):
    endNum = 0
    for i in range(img.shape[0]):
        img1 = img[i,:,:]
        # print(np.max(img1)
        if np.max(img1)>0 and endNum<5:
            pixelList.append(np.max(img1))
            img1[img1==255]=random.randint(20,255)
            img1[img1==0]=25
            im = Image.fromarray(img1)
            fileName = 'pic'+str(endNum)+'.jpeg'
            savePath = './image/'+fileName
            im.save(savePath)
            print(np.max(img1))
            # plt.imshow(img[i, :, :], cmap='gray')
            # plt.show()
            endNum=endNum+1
    # print(len(pixelList))
# niiFile = r'D:\1Gdata\mask\P00047619\mask_Bladder.nii.gz'
#
# itk_img = sitk.ReadImage(niiFile)
# img = sitk.GetArrayFromImage(itk_img)
# print(img.shape)  # (155, 240, 240) 表示各个维度的切片数量
# showNii(img)

# a = [[random.randint(1, 9) for j in range(1, 4)] for i in range(1, 4)]
# # print(a)
# b = [[random.randint(1, 4) for j in range(1, 4)] for i in range(1, 4)]
# # print(b)
# np1 = np.array(a)
# np2 = np.array(b)
# print(np1)
# print(np2)
# print(np.maximum(np1,np2))
# ds = pydicom.read_file(dicomFile)


## using simpleITK to load and save data.
import SimpleITK as sitk

path = "D:/1Gdata/mask/P00176954/image.nii.gz"
itk_img = sitk.ReadImage(path)
img = sitk.GetArrayFromImage(itk_img)
print("img shape:", img.shape)

## save
out = sitk.GetImageFromArray(img)
# # out.SetSpacing(itk_img.GetSpacing())
# # out.SetOrigin(itk_img.GetOrigin())
sitk.WriteImage(out, './util/image/img')


# 解决报错
# ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

#
# ds_array = ds.pixel_array
# new_image = ds.pixel_array.astype(float)
# scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
# scaled_image = np.uint8(scaled_image)
# # print(type(scaled_image))
# plt.imshow(scaled_image,cmap='gray')
# plt.show()
# plt.imshow(new_array,cmap='gray')
# final_image = Image.fromarray(scaled_image)
# final_image.show()



# ds_np = np.array(ds_array)

# maxPix = np.max(ds_np)
# print(ds_array.shape)
# x,y = ds_array.shape
# for row in range(x):
#     for col in range(y):
#         value = ds_array[row, col]
#         newValue = int((value/maxPix)*255)
#         ds_array[row][col]=newValue
        # if newValue>0:
        #     print(newValue)
        # print(value)


# for x in ds_array:
#     for y in x:
#         print(y)

# ds_np = np.array(ds_array)
# print(np.unique(ds_np))
# print(ds_array.shape)
# print(ds_array)
# plt.imshow(ds_array,cmap='gray')

# print(ds.dir())
# img2 = Image.open(path1)




# ds_array = ds.pixel_array
# plt.show(ds_array,cmap='gray')
# plt.figure(figsize=(10, 10))
# plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
# plt.show()



# print(st.size)
# print(st2.size)
# blendImg = Image.blend(st, st2, alpha = 0.5)
# blendImg.show()
# merge = Image.blend(st,st2,0.5)
# merge.save("mask_2.png")
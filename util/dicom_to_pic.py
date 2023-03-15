import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

# 数据
path = './dicom/'
file = 'p1.nii.gz'
# 目标路径
save_path = './dicom_to_img'
if not os.path.exists(save_path):
    os.mkdir(save_path)

itk_img = sitk.ReadImage(os.path.join(path, file))
data_np = sitk.GetArrayFromImage(itk_img)

# save extracted image
for idx in range(data_np.shape[0]):
    fname = str(idx).zfill(3) + '.png'
    # plt.imsave(os.path.join(save_path,fname), data_np[idx]) # 保存后像素0~255
    cv2.imwrite(os.path.join(save_path, fname), data_np[idx])  # 像素值为原始值(0,1,2,4)

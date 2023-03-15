import SimpleITK as sitk
import os

save_path = './img2'

file_list = os.listdir(save_path)
file_list.sort()
file_names = [os.path.join(save_path, f) for f in file_list]
# print(file_names)

newspacing = [1, 1, 1]  # 设置x，y, z方向的空间间隔
reader = sitk.ImageSeriesReader()
reader.SetFileNames(file_names)
vol = reader.Execute()
vol.SetSpacing(newspacing)
sitk.WriteImage(vol, './dicom/volume3.nii.gz')  # 保存为volume.nii.gz也可
import os

import SimpleITK as sitk
import cv2
import numpy as np
from PIL import Image


# 递归返回mask名字
def list_files_recursive(folder_path):
    nameList = []
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # print("文件:", os.path.join(root, file))
                if "image" in file:
                    nameList.append(file[0:-7])
                else:
                    nameList.append(file[5:-7])
            # for dir in dirs:
            #     print("文件夹:", os.path.join(root, dir))
        return nameList
    else:
        print(f"文件夹 '{folder_path}' 不存在。")


def count_names(names):
    name_counts = {}
    for name in names:
        if name in name_counts:
            name_counts[name] += 1
        else:
            name_counts[name] = 1
    # 字典序
    name_counts = sorted(name_counts.items(), key=lambda x: x[0], reverse=True)

    # 次数排序
    # name_counts = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
    return name_counts


# 判断一个文件夹里面是否拥有给定的多个文件
def fileInDir(folder_path, files_to_check):
    # = ["file1.txt", "file2.txt", "file3.txt"]

    all_files_exist = True

    for file_name in files_to_check:
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)
        if not os.path.exists(file_path):
            all_files_exist = False
            break
    return all_files_exist


# 给mask加前后缀
def addMaskInfo(inputImg):
    prefix = "mask_"
    suffix = ".nii.gz"
    modified_strings = []
    for string in inputImg:
        modified_string = prefix + string + suffix
        modified_strings.append(modified_string)
    # modified_strings.append("image.nii.gz")
    return modified_strings


# 返回有list数据的病人文件夹
def dataInDir(folder_path, files_to_check):
    # files_to_check = addMaskInfo(files_to_check)
    folder_names = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if
                    os.path.isdir(os.path.join(folder_path, name))]
    res = []
    for folder_name in folder_names:
        if fileInDir(folder_name, files_to_check):
            res.append(folder_name)
    return res


def generate_mask(folders_path, files_to_check, save_path, generate_type, axleStr):
    # generate_type：0 全生成   generate_type：1 只生成有像素点的
    # axle 0,1,2 三个轴
    axleDic = {"x": 0, "y": 1, "z": 2}
    generate_type = 0 if generate_type == "all" else 1
    axle = axleDic.get(axleStr)
    allImg = 0
    # files_to_check = addMaskInfo(files_to_check)
    for i, folder_path in enumerate(folders_path):
        npList = []
        name = folder_path.split('/')[-1]
        countImg = 0
        countAll = 0
        print("生成", name, "病人的数据中的", axleStr, "轴", i + 1, "/", len(folders_path))
        for j, file in enumerate(files_to_check):
            imgPath = os.path.join(folder_path, file)
            # gap = 30
            gap = 1
            pixel = gap * (j + 1)
            # print("fileName",imgPath,"pix",pixel)
            # print(imgPath)
            itk_img = sitk.ReadImage(imgPath)
            imgs = sitk.GetArrayFromImage(itk_img)
            for index in range(imgs.shape[axle]):
                if axle == 0:
                    img1 = imgs[index, :, :]
                elif axle == 1:
                    img1 = imgs[:, index, :]
                else:
                    img1 = imgs[:, :, index]
                if np.max(img1) > 0:
                    img1[img1 == 255] = pixel
                if len(npList) < imgs.shape[axle]:
                    npList.append(img1)
                else:
                    img2 = npList[index]
                    newImg = np.maximum(img1, img2)
                    npList[index] = newImg

        for j, img in enumerate(npList):
            fileName = name + '_' + str(j) + '_mask.png'
            savePathDir = os.path.join(save_path, axleStr)
            create_folder_if_not_exists(savePathDir)
            savePathFile = os.path.join(savePathDir, fileName)

            im = Image.fromarray(img)
            # print(np.max(im))
            countAll = countAll + 1
            if np.max(im) >= generate_type:
                # if np.max(im) > 0:
                countImg = countImg + 1
                allImg = allImg + 1
                im.save(savePathFile)
        # print("保存了",name,"病人",countAll,"张图片中的",countImg,"张图片")
        print("保存了", name, "病人", countImg, "张图片")
    print("保存成功,一共", allImg, "张图片")


def clahe_qu(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_res = np.zeros_like(imgs)
    for i in range(len(imgs)):
        img_res[i, :, :] = clahe.apply(np.array(imgs[i, :, :], dtype=np.uint8))
    return img_res


def generate_img(folders_path, save_path, axleStr):
    allImg = 0
    axleDic = {"x": 0, "y": 1, "z": 2}
    axle = axleDic.get(axleStr)
    for i, folder_path in enumerate(folders_path):
        # name = '1'
        name = folder_path.split('/')[-1]
        print("生成", name, "病人的数据中的", axleStr, "轴", i + 1, "/", len(folders_path))
        imagePath = os.path.join(folder_path, "image.nii.gz")
        itk_img = sitk.ReadImage(imagePath)
        imgs = sitk.GetArrayFromImage(itk_img)
        npList = []
        for index in range(imgs.shape[axle]):
            # img1 = imgs[index, :, :]
            if axle == 0:
                img1 = imgs[index, :, :]
            elif axle == 1:
                img1 = imgs[:, index, :]
            else:
                img1 = imgs[:, :, index]

            new_array = (np.maximum(img1, 0) / img1.max()) * 255.0
            new_array = np.uint8(new_array)
            npList.append(new_array)
            # im = Image.fromarray(new_array)
            # print(type(im))
            # fileName = name + '_' + str(index) + '.png'
        # print(imagePath)
        image_array = np.array(npList)
        img_clahe = clahe_qu(image_array)
        countImg = 0
        for index in range(imgs.shape[axle]):
            im = Image.fromarray(img_clahe[index])
            fileName = name + '_' + str(index) + '.png'

            savePathDir = os.path.join(save_path, axleStr)
            create_folder_if_not_exists(savePathDir)
            savePathFile = os.path.join(savePathDir, fileName)

            im.save(savePathFile)
            countImg = countImg + 1
            allImg = allImg + 1
        print("保存了", name, "病人", countImg, "张图片")
    print("保存成功,一共", allImg, "张图片")


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"已创建文件夹 '{folder_path}'")


if __name__ == '__main__':
    dataSource = "F:/unet_data/0917_27data/"  # 数据源文件夹

    save_train_path = "F:/unet_data/0917_14_7_train"  # 训练集 保存mask和img的位置
    save_val_path = "F:/unet_data/0917_14_7_val"  # 验证集 保存mask和img的位置

    create_folder_if_not_exists(save_train_path)
    create_folder_if_not_exists(save_val_path)
    # maskList = {"SpinalCord", "GTV-ln", "GTV-T", "Pelvis"}  # 要参与分割的部分,其余详见niiToImg.txt
    maskList = {"Pelvis", "Femoral-head-R", "Femoral-head-L", "Bladder", "GTV-ln",
                "GTV-T"}  # 要参与分割的部分,其余详见niiToImg.txt
    # maskList = {"GTV-ln", "GTV-T"}  # 要参与分割的部分,其余详见niiToImg.txt

    maskList = addMaskInfo(maskList)
    all_dir = dataInDir(dataSource, maskList)  # 符合要求的病人文件夹

    val_dir_count = 2

    train_dir = all_dir[0:-val_dir_count]
    val_dir = all_dir[-val_dir_count:]
    print("train",train_dir)
    print("val",val_dir)

    # print(input_dir)
    # input_dir = ["F:/unet_data/gz_data/P00179632"]  # 符合要求的病人文件夹
    # input_dir = ["F:/unet_data/gz_data/P00179632","F:/unet_data/gz_data/P00179663"]  # 符合要求的病人文件夹

    generate_mask(train_dir, maskList, save_train_path, "all", "x")  # 生成mask
    generate_mask(train_dir, maskList, save_train_path, "all", "y")  # 生成mask
    generate_mask(train_dir, maskList, save_train_path, "all", "z")  # 生成mask

    generate_img(train_dir, save_train_path, "x")  # 生成img
    generate_img(train_dir, save_train_path, "y")  # 生成img
    generate_img(train_dir, save_train_path, "z")  # 生成img

    generate_mask(val_dir, maskList, save_val_path, "all", "x")  # 生成mask
    generate_mask(val_dir, maskList, save_val_path, "all", "y")  # 生成mask
    generate_mask(val_dir, maskList, save_val_path, "all", "z")  # 生成mask

    generate_img(val_dir, save_val_path, "x")  # 生成img
    generate_img(val_dir, save_val_path, "y")  # 生成img
    generate_img(val_dir, save_val_path, "z")  # 生成img

    # 分析数据分布
    # print(input_dir, len(input_dir),imgList)
    # print(count_names(list_files_recursive(dataSource)))

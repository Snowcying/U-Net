import json
import os
import random

import cv2
import numpy as np


def select_random_elements2(array1, array2, percentage):
    # 确定要选择的元素数量
    num_elements = int(len(array1) * percentage)

    array1 = np.array(array1)
    array2 = np.array(array2)
    # 生成随机索引
    random_indices = np.random.choice(len(array1), size=num_elements, replace=False)

    # 从原始数组中选择相应的元素
    selected_elements_array1 = array1[random_indices].tolist()
    selected_elements_array2 = array2[random_indices].tolist()

    return selected_elements_array1, selected_elements_array2


def generate_file(folder_path, axle, type):
    data_img = []
    data_label = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if 'mask' in filename:
            data_label.append(img_path)
        else:
            data_img.append(img_path)
    img = []
    label = []
    for i in data_label:
        value = np.max(cv2.imread(i))
        try:
            if value == 0:
                # 不筛选
                # if value > 0:
                ranNum = random.random()
                # if ranNum>0.68:
                #     data_newlabel.append(i)
                #     i_img=i[:-9]+houzhui
                #     data_newimg.append(i_img)
            else:
                label.append(i)
                i_img = i[:-9] + '.png'
                img.append(i_img)
        except:
            pass
    random_img, random_mask = select_random_elements2(img, label, 1)
    print(len(img))
    print(len(label))
    jsonImg = json.dumps(random_img)
    # jsonImg = json.dumps(img)
    f1 = open('./temp/' + axle + '/img_' + type + '.txt', 'w')
    f1.write(jsonImg)
    f1.close()
    jsonMask = json.dumps(random_mask)
    # jsonMask = json.dumps(label)
    f2 = open('./temp/' + axle + '/mask_' + type + '.txt', 'w')
    f2.write(jsonMask)
    f2.close()


def get_file(axle, type):
    print("正在读取文件")
    f = open('./temp/' + axle + '/img_' + type + '.txt', 'r')
    jsonImg = f.read()
    img = json.loads(jsonImg)

    f = open('./temp/' + axle + '/mask_' + type + '.txt', 'r')
    jsonMask = f.read()
    label = json.loads(jsonMask)
    print("图片有", len(img), '张')
    return img, label


if __name__ == '__main__':
    # axle = "z"

    type = "train"

    dataSourceDir = 'F:/unet_data/0917_14_7_' + type + '/'

    dataSource = dataSourceDir + "x"
    generate_file(dataSource, "x", type)

    dataSource = dataSourceDir + "y"
    generate_file(dataSource, "y", type)

    dataSource = dataSourceDir + "z"
    generate_file(dataSource, "z", type)

    type = "val"

    dataSourceDir = 'F:/unet_data/0917_14_7_' + type + '/'

    dataSource = dataSourceDir + "x"
    generate_file(dataSource, "x", type)

    dataSource = dataSourceDir + "y"
    generate_file(dataSource, "y", type)

    dataSource = dataSourceDir + "z"
    generate_file(dataSource, "z", type)

    # img, label = get_file(axle)
    # print(label,len(label))

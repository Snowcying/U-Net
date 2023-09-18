import datetime
import os

import numpy as np

import niiToImg


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


def select_random_elements(array1, array2, percentage):
    # 确定要选择的元素数量
    num_elements = int(len(array1) * percentage)
    array1 = np.array(array1)
    array2 = np.array(array2)

    # 生成随机索引
    random_indices = np.random.choice(len(array1), size=num_elements, replace=False)

    # 从原始数组中选择相应的元素
    selected_elements_array1 = array1[random_indices]
    selected_elements_array2 = array2[random_indices]

    # 从剩余的数组中选择相应的元素
    remaining_indices = np.setdiff1d(np.arange(len(array1)), random_indices)
    remaining_elements_array1 = array1[remaining_indices]
    remaining_elements_array2 = array2[remaining_indices]

    return selected_elements_array1, selected_elements_array2, remaining_elements_array1, remaining_elements_array2


def test_select_random_elements():
    img = ["F:/unet_data/0915/x\\P00176954_100.png", "F:/unet_data/0915/x\\P00176954_101.png",
           "F:/unet_data/0915/x\\P00176954_102.png", "F:/unet_data/0915/x\\P00176954_103.png",
           "F:/unet_data/0915/x\\P00176954_104.png", "F:/unet_data/0915/x\\P00176954_105.png",
           "F:/unet_data/0915/x\\P00176954_106.png", "F:/unet_data/0915/x\\P00176954_107.png",
           "F:/unet_data/0915/x\\P00176954_108.png", "F:/unet_data/0915/x\\P00176954_109.png",
           "F:/unet_data/0915/x\\P00176954_10.png", "F:/unet_data/0915/x\\P00176954_110.png",
           "F:/unet_data/0915/x\\P00176954_111.png", "F:/unet_data/0915/x\\P00176954_112.png",
           "F:/unet_data/0915/x\\P00176954_113.png", "F:/unet_data/0915/x\\P00176954_114.png",
           "F:/unet_data/0915/x\\P00176954_115.png", "F:/unet_data/0915/x\\P00176954_116.png",
           "F:/unet_data/0915/x\\P00176954_117.png", "F:/unet_data/0915/x\\P00176954_118.png",
           "F:/unet_data/0915/x\\P00176954_119.png", "F:/unet_data/0915/x\\P00176954_11.png",
           "F:/unet_data/0915/x\\P00176954_120.png", "F:/unet_data/0915/x\\P00176954_121.png",
           "F:/unet_data/0915/x\\P00176954_122.png", "F:/unet_data/0915/x\\P00176954_123.png"]
    mask = ["F:/unet_data/0915/x\\P00176954_100_mask.png", "F:/unet_data/0915/x\\P00176954_101_mask.png",
            "F:/unet_data/0915/x\\P00176954_102_mask.png", "F:/unet_data/0915/x\\P00176954_103_mask.png",
            "F:/unet_data/0915/x\\P00176954_104_mask.png", "F:/unet_data/0915/x\\P00176954_105_mask.png",
            "F:/unet_data/0915/x\\P00176954_106_mask.png", "F:/unet_data/0915/x\\P00176954_107_mask.png",
            "F:/unet_data/0915/x\\P00176954_108_mask.png", "F:/unet_data/0915/x\\P00176954_109_mask.png",
            "F:/unet_data/0915/x\\P00176954_10_mask.png", "F:/unet_data/0915/x\\P00176954_110_mask.png",
            "F:/unet_data/0915/x\\P00176954_111_mask.png", "F:/unet_data/0915/x\\P00176954_112_mask.png",
            "F:/unet_data/0915/x\\P00176954_113_mask.png", "F:/unet_data/0915/x\\P00176954_114_mask.png",
            "F:/unet_data/0915/x\\P00176954_115_mask.png", "F:/unet_data/0915/x\\P00176954_116_mask.png",
            "F:/unet_data/0915/x\\P00176954_117_mask.png", "F:/unet_data/0915/x\\P00176954_118_mask.png",
            "F:/unet_data/0915/x\\P00176954_119_mask.png", "F:/unet_data/0915/x\\P00176954_11_mask.png",
            "F:/unet_data/0915/x\\P00176954_120_mask.png", "F:/unet_data/0915/x\\P00176954_121_mask.png",
            "F:/unet_data/0915/x\\P00176954_122_mask.png", "F:/unet_data/0915/x\\P00176954_123_mask.png"]

    # selected_array1, selected_array2, remaining_array1, remaining_array2 = select_random_elements(img, mask, 1)
    selected_array1, selected_array2 = select_random_elements2(img, mask, 1)

    print("选择的元素数组1:", selected_array1)
    print("选择的元素数组2:", selected_array2)
    # print("剩余的元素数组1:", remaining_array1)
    # print("剩余的元素数组2:", remaining_array2)


def timeUtil():
    now = datetime.datetime.now()
    # 格式化输出为字符串
    formatted_datetime = "{0}-{1}-{2}-{3}-{4}".format(now.year, now.month, now.day, now.hour, now.minute)

    # 打印结果
    print("格式化后的日期和时间：", formatted_datetime)
    return formatted_datetime


def test_timeUtil():
    str = timeUtil()
    return str


def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功！")


def count_names(names):
    name_counts = {}
    for name in names:
        if name in name_counts:
            name_counts[name] += 1
        else:
            name_counts[name] = 1
    # 字典序
    # name_counts = sorted(name_counts.items(), key=lambda x: x[0], reverse=True)

    # 次数排序
    name_counts = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
    return name_counts

if __name__ == '__main__':
    dataSource = "F:/unet_data/0917_27data/"
    maskList = {"Pelvis", "Femoral-head-R", "Femoral-head-L", "Bladder","GTV-ln",
                "GTV-T"}

    maskList = niiToImg.addMaskInfo(maskList)
    input_dir = niiToImg.dataInDir(dataSource, maskList)
    print(input_dir)

    nameList=niiToImg.list_files_recursive(dataSource)
    name_counts=count_names(nameList)
    for x,y in name_counts:
        if y>1:
         print(x,y,"次")
    # img,label=getFile.get_file()
    # print(img)
    # 示例用法
    # test_select_random_elements()
    # timeStr = test_timeUtil()
    # dir = os.path.join('.',timeStr,'x')
    # check_and_create_folder(dir)
    # print(dir)

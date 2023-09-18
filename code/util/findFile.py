import glob
import os

folder_path = "F:/unet_data/0917/"

search_file_name = "mask_"+"FemoralHead_R"+".nii.gz"   # from
replacement_file_name = "mask_"+"Femoral-head-R"+".nii.gz"  # to

# 递归地遍历文件夹及其子文件夹
count = 0
for root, dirs, files in os.walk(folder_path):
    # 在当前文件夹中搜索匹配的文件
    file_paths = glob.glob(os.path.join(root, search_file_name))

    if file_paths:
        print("找到以下文件：")
        for path in file_paths:
            print(path)
            # 获取文件所在的目录路径
            directory = os.path.dirname(path)
            # 构建新的文件路径
            new_path = os.path.join(directory, replacement_file_name)
            # 进行文件名替换
            os.rename(path, new_path)
            print("替换为：" + new_path)
            count=count+1
print("共找到 ",count)
    # else:
        # print("未找到匹配的文件。")

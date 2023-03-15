# 01改名加mask标签，合并到同一个文件夹
import os
# path = "D:/dataImage/image/P00047619/mask"
# des = "D:/dataImage/image/P00047619/test"

# path1为P000XXXX的父文件夹
num = 1
path1 = "D:/dataImage/imageAll/"
dirList = os.listdir(path1)
for dirName in dirList:
    pathMask = path1+dirName+'/mask'
    pathData = path1+dirName+'/data'
    for file in os.listdir(pathData):
        pic = pathData+'/'+file
        newName = file.split('.png')[0]+dirName+'.png'
        os.rename(os.path.join(pathData, file), os.path.join(pathData, newName))
    for file in os.listdir(pathMask):
        pic = pathMask+'/'+file
        newName = file.split('.png')[0]+dirName+'_mask'+'.png'
        os.rename(os.path.join(pathMask, file), os.path.join(pathData, newName))


# for file in os.listdir(path):
#     pic = path +'/'+file
#     newName = file.split('.png')[0]+'_mask'+'.png'
    # newName = file.split('.png')[0] + '.png'
    # print(os.path.exists(pic))
    # print(path+'/' + file)
    # print(newName)
    # os.rename(os.path.join(path, file), os.path.join(des, newName))
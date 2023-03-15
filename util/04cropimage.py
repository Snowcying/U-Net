from PIL import Image
import os

path = "D:/dataImage/finalAllData/"
savePath = "D:/dataImage/finalCropedData/1/"

for file in os.listdir(path):
    filePath = path+file
    # print(file)
    img = Image.open(filePath)
    region=img.crop((50,50,402,402))
    imgPath = savePath+file
    region.save(imgPath)




# img = Image.open(path)
# region = img.crop((50,50,380,380))
# region.save('out.png')
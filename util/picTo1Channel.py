import cv2 as cv
from PIL import Image

def image_pixel(image_path,save_dir):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    filename = image_path.split('/')[-1]
    save_path = save_dir+filename
    # cv.imshow('input', img)

    h, w, c = img.shape
    # 遍历像素点，修改图像b,g,r值
    for row in range(h):
        for col in range(w):
            b,g,r = img[row, col]
            x = 0
            if(b>x or g>x or r>x):
                img[row, col] = (255,255,255)
            else:
                # print(b,g,r)
                img[row,col]=(0,0,0)

    # cv.imshow('result', img)
    # cv.waitKey()
    cv.imwrite(save_path,img)
    return save_path


def threeChannleToOneChannel(save_path):
    image = Image.open(save_path)
    image = image.convert("L")
    image.save(save_path)

image_path='C:/Users/Administrator/Desktop/data/data3channel/1013data/axial0079_mask.png'
save_dir = 'C:/Users/Administrator/Desktop/data/test/'



def trans(image_path,save_dir):
    save_path = image_pixel(image_path, save_dir)
    threeChannleToOneChannel(save_path)

# trans(image_path,save_dir)



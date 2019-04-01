import cv2
import os
import sys
import numpy as np
from PIL import Image
import collections


def pHash(imgfile, size_height):
    """get image pHash value"""
    # 加载并调整图片为64x64灰度图片
    img = cv2.imread(imgfile, 0)
    img = cv2.resize(img, (size_height, size_height), interpolation=cv2.INTER_CUBIC)

        # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img       # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(size_height, size_height)

    # 把二维list变成一维list
    img_list = flatten(vis1.tolist())

    # 计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]), 2) for x in range(0, size_height*size_height, 4)])


def flatten(a):
    if not isinstance(a, (list, )):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
    return b


def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


def dHash(image, size_height):
    """
    计算图片的dHash值
    :param image: PIL.Image
    :return: dHash值,string类型
    """
    image = Image.open(image)
    difference = __difference(image, size_height)
    # 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
    return ''.join(['%x' % int(''.join(difference[x:x + 4]), 2) for x in range(0, size_height * size_height, 4)])


def __difference(image, size_height):
    """
    *Private method*
    计算image的像素差值
    :param image: PIL.Image
    :return: 差值数组。0、1组成
    """
    resize_width = size_height + 1
    resize_height = size_height
    # 1. resize to (32,33)
    smaller_image = image.resize((resize_width, resize_height))
    # 2. 灰度化 Grayscale
    grayscale_image = smaller_image.convert("L")
    # 3. 比较相邻像素
    pixels = list(grayscale_image.getdata())
    difference = []
    for row in range(resize_height):
        row_start_index = row * resize_width
        for col in range(resize_width - 1):
            left_pixel_index = row_start_index + col
            difference.append('1' if pixels[left_pixel_index] > pixels[left_pixel_index + 1] else '0')
    return difference


size_height = 32
pHASH1 = pHash('D:\\Github\\PreprocessedData\\data\\test01.png', size_height)
pHASH2 = pHash('D:\\Github\\PreprocessedData\\data\\test02.png', size_height)
pHASH3 = pHash('D:\\Github\\PreprocessedData\\data\\test03.png', size_height)
pHASH4 = pHash('D:\\Github\\PreprocessedData\\data\\_2000_sampled_12_6.png', size_height)

dHASH1 = dHash('D:\\Github\\PreprocessedData\\data\\test01.png', size_height)
dHASH2 = dHash('D:\\Github\\PreprocessedData\\data\\test02.png', size_height)
dHASH3 = dHash('D:\\Github\\PreprocessedData\\data\\test03.png', size_height)
dHASH4 = dHash('D:\\Github\\PreprocessedData\\data\\_2000_sampled_12_6.png', size_height)

p_D_1 = hammingDist(pHASH1, pHASH2)
p_D_2 = hammingDist(pHASH1, pHASH3)
p_D_3 = hammingDist(pHASH1, pHASH4)
d_D_1 = hammingDist(dHASH1, dHASH2)
d_D_2 = hammingDist(dHASH1, dHASH3)
d_D_3 = hammingDist(dHASH1, dHASH4)

p_out_score_1 = 1 - p_D_1 * 1. / (size_height * size_height / 4)
p_out_score_2 = 1 - p_D_2 * 1. / (size_height * size_height / 4)
p_out_score_3 = 1 - p_D_3 * 1. / (size_height * size_height / 4)

d_out_score_1 = 1 - d_D_1 * 1. / (size_height * size_height / 4)
d_out_score_2 = 1 - d_D_2 * 1. / (size_height * size_height / 4)
d_out_score_3 = 1 - d_D_3 * 1. / (size_height * size_height / 4)


print(p_out_score_1, p_out_score_2, p_out_score_3, '||', p_D_1, p_D_2, p_D_3)
print(d_out_score_1, d_out_score_2, d_out_score_3, '||', d_D_1, d_D_2, d_D_3)

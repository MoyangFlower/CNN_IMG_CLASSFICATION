import cv2
import os
import sys
import time
import numpy as np
from PIL import Image
import collections
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

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


def _areas_of_triangles(a, bs, c):
    """Calculate areas of triangles from duples of vertex coordinates.
    Uses implicit numpy broadcasting along first axis of ``bs``.
    Returns
    -------
    numpy.array
        Array of areas of shape (len(bs),)
    """
    bs_minus_a = bs - a
    a_minus_bs = a - bs
    return 0.5 * abs((a[0] - c[0]) * (bs_minus_a[:, 1]) - (a_minus_bs[:, 0]) * (c[1] - a[1]))


def downsample(data, n_out):
    """Downsample ``data`` to ``n_out`` points using the LTTB algorithm.

    Reference
    ---------
    Sveinn Steinarsson. 2013. Downsampling Time Series for Visual
    Representation. MSc thesis. University of Iceland.

    Constraints
    -----------
      - ncols(data) == 2
      - 3 <= n_out <= nrows(data)
      - ``data`` should be sorted on the first column.

    Returns
    -------
    numpy.array
        Array of shape (n_out, 2)
    """
    # Validate input
    if data.shape[1] != 2:
        raise ValueError('data should have 2 columns')

    if np.any(data[1:, 0] <= data[:-1, 0]):
        raise ValueError('data should be sorted on first column')

    if n_out > data.shape[0]:
        raise ValueError('n_out must be <= number of rows in data')

    if n_out == data.shape[0]:
        return data

    if n_out < 3:
        raise ValueError('Can only downsample to a minimum of 3 points')

    start = time.time()

    # Split data into bins
    n_bins = n_out - 2
    data_bins = np.array_split(data[1: len(data) - 1], n_bins)

    # Prepare output array
    # First and last points are the same as in the input.
    out = np.zeros((n_out, 2))
    out[0] = data[0]
    out[len(out) - 1] = data[len(data) - 1]

    # Largest Triangle Three Buckets (LTTB):
    # In each bin, find the point that makes the largest triangle
    # with the point saved in the previous bin
    # and the centroid of the points in the next bin.
    for i in range(len(data_bins)):
        this_bin = data_bins[i]

        if i < n_bins - 1:
            next_bin = data_bins[i + 1]
        else:
            next_bin = data[len(data) - 1:]

        a = out[i]
        bs = this_bin
        c = next_bin.mean(axis=0)
        next_bin.std()
        areas = _areas_of_triangles(a, bs, c)

        out[i + 1] = bs[np.argmax(areas)]

    end = time.time()
    print("LTTB cost time: ", end-start)
    return out, end-start


def test_downsampling():
    # csv = 'tests/timeseries.csv'
    # data = np.genfromtxt(csv, delimiter=',', names=True)
    # xs = data['X']
    # ys = data['Y']
    # data = np.array([xs, ys]).T
    # out = downsample(data, 100)
    # assert out.shape == (100, 2)
    pass


def downsample_avg(data, n_out):
    # Validate input
    if data.shape[1] != 2:
        raise ValueError('data should have 2 columns')

    if np.any(data[1:, 0] <= data[:-1, 0]):
        raise ValueError('data should be sorted on first column')

    if n_out > data.shape[0]:
        raise ValueError('n_out must be <= number of rows in data')

    if n_out == data.shape[0]:
        return data

    if n_out < 3:
        raise ValueError('Can only downsample to a minimum of 3 points')

    start = time.time()
    # Split data into bins
    n_bins = n_out - 2
    data_bins = np.array_split(data[1: len(data) - 1], n_bins)

    # Prepare output array
    # First and last points are the same as in the input.
    out = np.zeros((n_out, 2))
    out[0] = data[0]
    out[len(out) - 1] = data[len(data) - 1]

    # average:
    # In each bin, find the average point in this bin

    for i in range(len(data_bins)):
        out[i + 1] = data_bins[i].mean(axis=0)

    end = time.time()
    print("avg cost time: ", end-start)
    return out, end-start


if __name__ == '__main__':

    Y = []
    X = []
    size_W = 12
    size_H = 4
    X_index = 0
    num_point_arr = [500, 10000, 50000, 100000, 200000]
    sample_rate = 0.1
    time_y1 = []
    time_y2 = []
    save_img_path = "D:\\Github\\PreprocessedData\\paper img\\time-consuming.png"
    for num_point in num_point_arr:
        with open("D:\\Github\\PreprocessedData\\down-sampling data\\3414.txt") as f:
            lines = f.readlines()
            lines.sort()
            for line in lines:
                if line.startswith('startdate'):
                    continue
                values = line.split('|')[-1]
                for value in values.split(" "):
                    if 'NULL' in value:
                        value = 0
                    temp = value
                    # if float(value.rstrip("\n")) > 500 or float(value.rstrip("\n")) < -500:
                    #     continue
                    X.append(X_index)
                    X_index += 1

                    Y.append(float(value.rstrip("\n")))

                    if num_point == 0:
                        continue
                    if len(Y) > num_point:
                        break
                if num_point == 0:
                    continue
                if len(Y) > num_point:
                    break
        raw_data = np.array([X, Y]).T
        print(len(X), "\n", len(Y))

        LTTB_sample, y1_t = downsample(raw_data, n_out=int(len(X) * sample_rate))
        avg_sample, y2_t = downsample_avg(raw_data, n_out=int(len(X) * sample_rate))
        time_y1.append(y1_t/y2_t)
        # time_y2.append(y2_t)

    plt.figure(figsize(size_W, size_H))
    plt.plot(num_point_arr, time_y1, label="LTTB/avg", linewidth=0.5, marker="x", linestyle="-")
    # plt.plot(num_point_arr, time_y2, label="avg", linewidth=0.5, linestyle="-")
    plt.legend(loc='upper left')
    plt.title("Time-consuming times graph of LTTB and avg downsampling with data volume growth")
    plt.xlabel("number of raw point")
    plt.xticks(num_point_arr, [r'500', r'10000', r'50000', r'100000', r'200000'])
    plt.ylabel("times")
    plt.yticks(np.linspace(1, 10, 5))
    plt.savefig(save_img_path, dpi=1200)
    plt.show()

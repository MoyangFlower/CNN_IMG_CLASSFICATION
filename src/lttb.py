import numpy as np
import os
from PIL import Image
import cv2
import similaritymeasures
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from scipy.spatial.distance import euclidean
from dtw import dtw
from fastdtw import fastdtw
from scipy.integrate import simps
import sys
import time
from matplotlib.ticker import FuncFormatter
sys.setrecursionlimit(100000)


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


def downsampled(algorithm, data, n_out):
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

    if algorithm == "avg":
        # average:
        # In each bin, find the average point in this bin

        for i in range(len(data_bins)):
            out[i + 1] = data_bins[i].mean(axis=0)
    elif algorithm == "max":
        # max:
        # In each bin, find the max point in this bin
        for i in range(len(data_bins)):
            out[i + 1] = data_bins[i].max(axis=0)
    elif algorithm == "min":
        # max:
        # In each bin, find the max point in this bin
        for i in range(len(data_bins)):
            out[i + 1] = data_bins[i].min(axis=0)
    elif algorithm == "mid":
        # max:
        # In each bin, find the max point in this bin
        for i in range(len(data_bins)):
            out[i + 1] = data_bins[i][len(data_bins[i])//2]
    elif algorithm == "lttb":
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
    print("%s cost time: " % algorithm, end-start)
    return out


if __name__ == "__main__":

    item = "3414"
    txt_path = "D:\\Github\\PreprocessedData\\down-sampling data\\%s.txt" % item
    sample_img_path = "D:\\Github\\PreprocessedData\\down-sampling img\\"
    img_path = "D:\\Github\\PreprocessedData\\paper_img_3\\"
    Y = []
    X = []
    # 6 * 4
    Is_calculate_distance = True
    plot_complete_flag = 3
    size_height = 1024
    y_value = [[] for _ in range(5)]

    if Is_calculate_distance:
        size_W = 12
        size_H = 8
        save_flag = False
        dpi = 1200
    else:
        size_W = 9
        size_H = 6
        dpi = 800
        save_flag = True
    num_point = 200000
    sample_rate = 0.1
    min_y = -1000
    max_y = 1000
    ticks = 5
    X_index = 0
    temp = ''
    txt_name = txt_path.split("\\")[-1].replace(".txt", "")
    save_img_path = img_path + '%s_%s_%s_sampled_%s_%s.png' % (item, str(num_point), str(sample_rate), str(size_W), str(size_H))
    with open(txt_path) as f:
        lines = f.readlines()
        lines.sort()
        for line in lines:
            if line.startswith('startdate'):
                continue
            values = line.split('|')[-1]
            for value in values.split(" "):
                if 'NULL' in value:
                    value = temp
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
    print(len(X), len(Y))
    raw_data = np.array([X, Y]).T

    lttb_sample = downsampled("lttb", raw_data, n_out=int(len(X) * sample_rate))
    avg_sample = downsampled("avg", raw_data, n_out=int(len(X) * sample_rate))
    max_sample = downsampled("max", raw_data, n_out=int(len(X) * sample_rate))
    min_sample = downsampled("min", raw_data, n_out=int(len(X) * sample_rate))
    mid_sample = downsampled("mid", raw_data, n_out=int(len(X) * sample_rate))

    print("down sample done")

    if Is_calculate_distance:

        # raw_area = simps(Y, X, dx=0.001)
        # lttb_area = raw_area - simps(LTTB_sample[:, 0], LTTB_sample[:, 1], dx=0.001)
        # avg_area = raw_area - simps(avg_sample[:, 0], avg_sample[:, 1], dx=0.001)

        # 计算两条曲线之间的面积
        # lttb_area = similaritymeasures.area_between_two_curves(LTTB_sample, raw_data)
        # avg_area = similaritymeasures.area_between_two_curves(avg_sample, raw_data)

        # 计算量条曲线的dtw值
        # lttb_dtw, lttb_d = similaritymeasures.dtw(LTTB_sample, raw_data)
        # avg_dtw, avg_d = similaritymeasures.dtw(avg_sample, raw_data)
        #
        # print('LTTB_sampled:', lttb_area, lttb_dtw)
        # print('avg_sampled:', avg_area, avg_dtw)

        # 画出路径图
        # LTTB_distance, lttb_cost_matrix, lttb_acc_cost_matrix, lttb_path = dtw(LTTB_sample[:, 1], raw_data[:, 1],  dist=euclidean)
        # avg_distance, avg_cost_matrix, avg_acc_cost_matrix, avg_path = dtw(avg_sample[:, 1], raw_data[:, 1],  dist=euclidean)
        # plt.imshow(lttb_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        # plt.plot(lttb_path[0], lttb_path[1], 'w')
        # plt.show()
        # plt.imshow(avg_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        # plt.plot(avg_path[0], avg_path[1], 'w')
        # plt.show()

        # Discrete Frechet distance
        # print("calculate Discrete Frechet distance...")
        # df_lttb = similaritymeasures.frechet_dist(lttb_sample, raw_data)
        # df_avg = similaritymeasures.frechet_dist(avg_sample, raw_data)
        # df_max = similaritymeasures.frechet_dist(max_sample, raw_data)
        # df_min = similaritymeasures.frechet_dist(min_sample, raw_data)
        # df_mid = similaritymeasures.frechet_dist(mid_sample, raw_data)
        # print("df_lttb_sampled:%4.0f\n df_avg_sampled:%4.0f\n df_max_sampled:%4.0f\n "
        #       "df_min_sampled:%4.0f\n df_mid_sampled:%4.0f\n"
        #       % (df_lttb, df_avg, df_max, df_min, df_mid))
        #
        # # 快速dtw，euclidean欧氏距离
        # print("calculate dwt distance...")
        # dtw_lttb, path = fastdtw(lttb_sample, raw_data, dist=euclidean)
        # dtw_avg, path1 = fastdtw(avg_sample, raw_data, dist=euclidean)
        # dtw_max, path2 = fastdtw(max_sample, raw_data, dist=euclidean)
        # dtw_min, path3 = fastdtw(min_sample, raw_data, dist=euclidean)
        # dtw_mid, path4 = fastdtw(mid_sample, raw_data, dist=euclidean)
        # print("dtw_lttb_sampled:%4.0f\n dtw_avg_sampled:%4.0f\n dtw_max_sampled:%4.0f\n "
        #       "dtw_min_sampled:%4.0f\n dtw_mid_sampled:%4.0f\n"
        #       % (dtw_lttb, dtw_avg, dtw_max, dtw_min, dtw_mid))

        # hamming distance距离
        print("calculate hamming distance...")

        def plot_save_single(data, color, png_name):

            plt.plot(data[:, 0], data[:, 1], color=color, linewidth=0.2)
            # plt.yticks(np.linspace(min_y, max_y, ticks))
            # plt.legend(loc='upper left')
            # plt.title("raw_data")
            plt.savefig(img_path + txt_name + "%s.png" % png_name, dpi=dpi)
            plt.cla()
        if not os.path.exists(img_path + txt_name + "mid.png"):
            plot_save_single(raw_data, 'r', "raw")
            plot_save_single(lttb_sample, 'black', "lttb")
            plot_save_single(avg_sample, 'black', "avg")
            plot_save_single(max_sample, 'black', "max")
            plot_save_single(min_sample, 'black', "min")
            plot_save_single(mid_sample, 'black', "mid")

        pHASH1 = pHash(img_path + txt_name + "raw.png", size_height)
        pHASH2 = pHash(img_path + txt_name + "LTTB.png", size_height)
        pHASH3 = pHash(img_path + txt_name + "avg.png", size_height)
        pHASH4 = pHash(img_path + txt_name + "max.png", size_height)
        pHASH5 = pHash(img_path + txt_name + "min.png", size_height)
        pHASH6 = pHash(img_path + txt_name + "mid.png", size_height)

        hd_1_2 = hammingDist(pHASH1, pHASH2)
        hd_1_3 = hammingDist(pHASH1, pHASH3)
        hd_1_4 = hammingDist(pHASH1, pHASH4)
        hd_1_5 = hammingDist(pHASH1, pHASH5)
        hd_1_6 = hammingDist(pHASH1, pHASH6)

        print("pd_lttb_sampled:%4.0f\n pd_avg_sampled:%4.0f\n pd_max_sampled:%4.0f\n "
              "pd_min_sampled:%4.0f\n pd_mid_sampled:%4.0f\n"
              % (hd_1_2, hd_1_3, hd_1_4, hd_1_5, hd_1_6))
        # y_value[0].append(p_D_1_2)
        # y_value[1].append(p_D_1_3)
        # y_value[2].append(p_D_1_4)
        # y_value[3].append(p_D_1_5)
        # y_value[4].append(p_D_1_6)

    plt.figure(figsize(size_W, size_H))  # 按照指定比例生成图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if plot_complete_flag == 1:
        plt.subplot(3, 1, 1)
        plt.plot(raw_data[:, 0], raw_data[:, 1], label="raw_data", color='r', linewidth=0.2)
        plt.plot(lttb_sample[:, 0], lttb_sample[:, 1], label='LTTB_sampled', color='b', linewidth=0.2)
        plt.plot(avg_sample[:, 0], avg_sample[:, 1], label='avg_sampled', color='black', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("LTTB avg down-sampling")

        # plt.rcParams['savefig.dpi'] = 300
        # plt.rcParams['figure.dpi'] = 300

        plt.subplot(3, 2, 3)
        plt.plot(raw_data[:, 0], raw_data[:, 1], label="raw_data", color='r', linewidth=0.2)
        plt.plot(lttb_sample[:, 0], lttb_sample[:, 1], label='LTTB_sampled', color='b', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("LTTB down-sampling")

        plt.subplot(3, 2, 4)
        plt.plot(raw_data[:, 0], raw_data[:, 1], label="raw_data", color='r', linewidth=0.2)
        plt.plot(avg_sample[:, 0], avg_sample[:, 1], label='avg_sampled', color='black', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("avg down-sampling")

        plt.subplot(3, 3, 7)
        plt.plot(raw_data[:, 0], raw_data[:, 1], label="raw_data", color='r', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("raw_data")

        plt.subplot(3, 3, 8)
        plt.plot(lttb_sample[:, 0], lttb_sample[:, 1], label='LTTB_sampled', color='blue', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("LTTB")

        plt.subplot(3, 3, 9)
        plt.plot(avg_sample[:, 0], avg_sample[:, 1], label='avg_sampled', color='black', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("avg")

        if save_flag:
            plt.savefig(save_img_path, dpi=dpi)

    elif plot_complete_flag == 2:
        def plot_save_single(data, color, png_name):

            plt.plot(data[:, 0], data[:, 1], color=color, linewidth=0.2)
            plt.yticks(np.linspace(min_y, max_y, ticks))
            # plt.legend(loc='upper left')
            # plt.title("raw_data")
            plt.savefig(img_path + txt_name + "%s.png" % png_name, dpi=dpi)
            plt.cla()
        plot_save_single(raw_data, 'r', "raw")
        plot_save_single(lttb_sample, 'black', "lttb")
        plot_save_single(avg_sample, 'black', "avg")
        plot_save_single(max_sample, 'black', "max")
        plot_save_single(min_sample, 'black', "min")
        plot_save_single(mid_sample, 'black', "mid")

    elif plot_complete_flag == 3:

        def formatnum(x, pos):
            return '$%.1f$x$10^{4}$' % (x / 10000)
        formatter = FuncFormatter(formatnum)

        def plot_save_multi(x, y, num, data, color, title, xlable):

            plt.subplot(x, y, num)
            plt.plot(data[:, 0], data[:, 1], color=color, linewidth=0.2)
            plt.xticks([])
            plt.yticks(([]))
            # plt.yticks(np.linspace(min_y, max_y, ticks))
            plt.title(title)
            # plt.ylabel("value")
            plt.xlabel(xlable)

        plot_save_multi(2, 3, 1, raw_data, 'black', "", "(a) 原始折线图")
        plot_save_multi(2, 3, 2, lttb_sample, 'black', "", "(b) LTTB算法降采样后折线图")
        plot_save_multi(2, 3, 3, avg_sample, 'black', "", "(c) 均值算法降采样后折线图")
        plot_save_multi(2, 3, 4, max_sample, 'black', "", "(d)最大值算法降采样后折线图")
        plot_save_multi(2, 3, 5, min_sample, 'black', "", "(e)最小值算法降采样后折线图")
        plot_save_multi(2, 3, 6, mid_sample, 'black', "", "(f)中位数算法降采样后折线图")

        if save_flag:
            plt.savefig(save_img_path, dpi=dpi)
    plt.show()


from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import numpy as np

# plt.figure()
# plt.figure(figsize(16, 6))
# plt.plot(range(0, 10), [np.random.random(10) for i in range(0, 10)], label="exp_data", color='r', linewidth=2)
# plt.plot(range(0, 10), [np.random.random(10) for i in range(0, 10)], label='num_data', color='g', linewidth=2)
# # plt.plot(range(0, len(dwell)), dwell, label='dwell', color='b', linewidth=2)
# plt.legend(loc='upper left')
# plt.title("plot test")
# plt.savefig('test.png')
# plt.show()
# print("aaa&bbb".split(" ")[0].split("&")[1])


import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# data_file = "D:\\Github\\PreprocessedData\\data\\040013411201.txt"
#
# dataarr = np.loadtxt(data_file, str, comments=',')
#
# print(dataarr[:, 0])
# sortrow = dataarr[:, 0]
# dataarr = dataarr[dataarr[:, 0].argsort()]
# print(dataarr)

# Y = []
# X = []
# with open("D:\\Github\\PreprocessedData\\data\\040013411201.txt") as f:
#     dataarr = np.loadtxt(data_file, str, comments=',')
#     sortrow = dataarr[:, 0]
#     dataarr = dataarr[dataarr[:, 0].argsort()]
#     print(dataarr)
#     index = 0
#     lines = f.readlines()
#     lines.sort()
#     print(lines)
#     for line in lines:
#         # print(len(line.split(",")[-1].split(" ")))
#         for key, value in enumerate(line.split(",")[-1].split(" ")):
#             Y.append(float(value.rstrip("\n")))
# print(X, "\n", Y)
# for x in range(len(Y)):
#     X.append(x)
#
# raw_data = np.array([X, Y]).T


# 计算方差
def getss(list):
    # 计算平均值
    avg = sum(list)/len(list)
    # 定义方差变量ss，初值为0
    ss = 0
    for l in list:
        ss += (l-avg)*(l-avg)/len(list)
    return ss


# 获取每行像素平均值
def getdiff(img):
    # 定义高度
    Height = img.shape[0]

    # 灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avglist=[]
    # 计算每行均值，保存到avglist列表
    for i in range(Height):
        avg = sum(gray[i])/len(gray[i])
        avglist.append(avg)
    return avglist


if __name__ == '__main__':

    # 读取测试图片
    img_1 = cv2.imread("D:\\Github\\PreprocessedData\\data\\test01.png")
    diff_1 = getdiff(img_1)

    # 读取测试图片
    img_2 = cv2.imread("D:\\Github\\PreprocessedData\\data\\test02.png")
    diff_2 = getdiff(img_2)

    # 读取测试图片
    img_3 = cv2.imread("D:\\Github\\PreprocessedData\\data\\test03.png")
    diff_3 = getdiff(img_3)
    print('ss of raw->avg-sampling:', abs(getss(diff_1) - getss(diff_3)))
    print('ss of raw->LTTB-sampling:', abs(getss(diff_1) - getss(diff_2)))

    x = range(img_3.shape[0])
    plt.figure()
    plt.plot(x, diff_1, label="raw data")
    plt.plot(x, diff_2, label="$LTTB$")
    plt.plot(x, diff_3, label="$avg$")
    plt.title("ss plot")
    plt.legend()
    plt.show()
    plt.savefig("D:\\Github\\PreprocessedData\\data\\ss_plot.png")


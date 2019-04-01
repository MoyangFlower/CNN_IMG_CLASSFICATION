import numpy as np
import similaritymeasures
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from scipy.spatial.distance import euclidean
from dtw import dtw
from fastdtw import fastdtw
from scipy.integrate import simps
import sys
sys.setrecursionlimit(10000)


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

    return out


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

    return out


if __name__ == "__main__":

    txt_path = "D:\\Github\\PreprocessedData\\data\\040033411101.txt"
    Y = []
    X = []
    # 6 * 4
    Is_test_program = True
    plot_complete_flag = False
    if Is_test_program:
        size_W = 12
        size_H = 8
        save_flag = False
        dpi = 600
    else:
        size_W = 40
        size_H = 30
        dpi = 800
        save_flag = True
    num_point = 1000
    sample_rate = 0.3

    save_path =txt_path.replace('.txt', '_' + str(num_point)+'_' + str(sample_rate) + '_sampled_'+str(size_W) + '_'+str(size_H) + '.png')
    with open(txt_path) as f:
        lines = f.readlines()
        lines.sort()
        index = 0
        for line in lines[10:]:
            print(line.split(",")[0])
            # print(len(line.split(",")[-1].split(" ")))
            for key, value in enumerate(line.split(",")[-1].split(" ")):
                if "NULL" in value:
                    continue
                Y.append(float(value.rstrip("\n")))
                if len(Y) > num_point:
                    break
            if len(Y) > num_point:
                break
    # print(X, "\n", Y)
    for x in range(len(Y)):
        X.append(x)

    raw_data = np.array([X, Y]).T

    # data = np.array([range(1000), np.random.random(1000)]).T
    # print(raw_data[40:])

    LTTB_sample = downsample(raw_data, n_out=int(len(X) * sample_rate))
    avg_sample = downsample_avg(raw_data, n_out=int(len(X) * sample_rate))

    if Is_test_program:

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

        # 快速dtw，euclidean欧氏距离
        LTTB_distance, path = fastdtw(LTTB_sample, raw_data, dist=euclidean)
        avg_distance, path1 = fastdtw(avg_sample, raw_data, dist=euclidean)
        print('LTTB_sampled:', LTTB_distance)
        print('avg_sampled:', avg_distance)

    plt.figure(figsize(size_W, size_H))  # 按照指定比例生成图

    if plot_complete_flag:
        plt.subplot(3, 1, 1)
        plt.plot(raw_data[:, 0], raw_data[:, 1], label="raw_data", color='r', linewidth=0.2)
        plt.plot(LTTB_sample[:, 0], LTTB_sample[:, 1], label='LTTB_sampled', color='b', linewidth=0.2)
        plt.plot(avg_sample[:, 0], avg_sample[:, 1], label='avg_sampled', color='black', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("LTTB avg down-sampling")

        # plt.rcParams['savefig.dpi'] = 300
        # plt.rcParams['figure.dpi'] = 300

        plt.subplot(3, 2, 3)
        plt.plot(raw_data[:, 0], raw_data[:, 1], label="raw_data", color='r', linewidth=0.2)
        plt.plot(LTTB_sample[:, 0], LTTB_sample[:, 1], label='LTTB_sampled', color='b', linewidth=0.2)
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
        plt.plot(LTTB_sample[:, 0], LTTB_sample[:, 1], label='LTTB_sampled', color='blue', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("LTTB")

        plt.subplot(3, 3, 9)
        plt.plot(avg_sample[:, 0], avg_sample[:, 1], label='avg_sampled', color='black', linewidth=0.2)
        plt.legend(loc='upper left')
        plt.title("avg")
    else:

        plt.plot(raw_data[:, 0], raw_data[:, 1], color='r', linewidth=0.2)
        plt.yticks(np.linspace(0.18, 0.28, 5))
        # plt.legend(loc='upper left')
        # plt.title("raw_data")
        plt.savefig("D:\\Github\\PreprocessedData\\data\\test01.png", dpi=dpi)
        plt.cla()

        plt.plot(LTTB_sample[:, 0], LTTB_sample[:, 1], color='blue', linewidth=0.2)
        plt.yticks(np.linspace(0.18, 0.28, 5))
        # plt.legend(loc='upper left')
        # plt.title("LTTB")
        plt.savefig("D:\\Github\\PreprocessedData\\data\\test02.png", dpi=dpi)
        plt.cla()

        plt.plot(avg_sample[:, 0], avg_sample[:, 1], color='black', linewidth=0.2)
        plt.yticks(np.linspace(0.18, 0.28, 5))
        # plt.legend(loc='upper left')
        # plt.title("avg")
        plt.savefig("D:\\Github\\PreprocessedData\\data\\test03.png", dpi=dpi)
        plt.cla()
    if save_flag:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


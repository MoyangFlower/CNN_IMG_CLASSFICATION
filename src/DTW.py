import numpy as np
import similaritymeasures
import matplotlib.pyplot as plt
import sys
# from lttb import lttb
sys.setrecursionlimit(10000000)


X_1 = np.linspace(0, 500, 30, endpoint=True)
X_2 = np.linspace(0, 500, 20, endpoint=True)
C, S = np.cos(X_1), np.cos(X_2)
# plt.plot(X_1, C)
# plt.plot(X_2, S)
# plt.show()

# Generate random experimental data

exp_data = np.zeros((30, 2))
exp_data[:, 0] = X_1
exp_data[:, 1] = C

# Generate random numerical data

num_data = np.zeros((20, 2))
num_data[:, 0] = X_2
num_data[:, 1] = S

exp_data = np.array([range(3600), np.random.random(3600)]).T
num_data = np.array([range(300), np.random.random(300)]).T


# quantify the difference between the two curves using PCM
# pcm = similaritymeasures.pcm(exp_data, num_data)

# quantify the difference between the two curves using
# Discrete Frechet distance
df = similaritymeasures.frechet_dist(exp_data, num_data)

# quantify the difference between the two curves using
# area between two curves
# area = similaritymeasures.area_between_two_curves(exp_data, num_data)

# quantify the difference between the two curves using
# Curve Length based similarity measure
# cl = similaritymeasures.curve_length_measure(exp_data, num_data)

# quantify the difference between the two curves using
# Dynamic Time Warping distance
# dtw, d = similaritymeasures.dtw(exp_data, num_data)

# print the results
# print(pcm, df, area, cl, dtw)
print(df)

# plot the data
# plt.figure()
# plt.plot(exp_data[:, 0], exp_data[:, 1])
# plt.plot(num_data[:, 0], num_data[:, 1])
# plt.show()


# plt.figure()
# plt.plot(range(0, len(exp_data)), exp_data[:, 1], label="exp_data", color='r', linewidth=2)
# plt.plot(range(0, len(num_data)), num_data[:, 1], label='num_data', color='g', linewidth=2)
# # plt.plot(range(0, len(dwell)), dwell, label='dwell', color='b', linewidth=2)
# plt.legend(loc='upper left')
# plt.title("plot test")
# plt.show()

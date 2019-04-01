import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
x = np.array([2, 0, 1, 3, 2, 4, 2, 4, 2, 0, 2, 0]).reshape(-1, 1)
y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)




euclidean_norm = lambda x, y: np.abs(x - y)

d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

# print(d)
# >>> 0.1111111111111111 # Only the cost for the insertions is kept

# You can also visualise the accumulated cost and the shortest path
# P = [[1, 2], [2, 0], [2, 1], [3, 3], [4, 2], [5, 4], [6, 2], [7, 4], [8, 2], [9, 0], [10, 2], [11, 0]]
# Q = [[2, 1], [0, 1], [2, 2], [3, 4], [4, 2], [5, 1], [6, 2], [7, 0]]
# print(frdist(P, Q))


print(d, cost_matrix, acc_cost_matrix, path)
plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()


import dtw
import numpy as np
import matplotlib.pyplot as plt
# from numpy.linalg import norm

x = np.array(list(range(15, 20))).reshape(-1, 1)
y = np.array(list(range(5, 10))).reshape(-1, 1)


dist, cost, acc, path = dtw.fastdtw(x, y, dist=lambda x, y: abs(x-y))
dist
plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, acc.shape[0]-0.5))
plt.ylim((-0.5, acc.shape[1]-0.5))
plt.plot(list(range(len(x))), x)
plt.plot(y, list(range(len(y))))
plt.show()
# 这个dtw距离似乎会随着时间序列的长度改变啊！这个可咋整呢？
# 不过经过实验之后发现，虽然会变，但是还是会收敛的

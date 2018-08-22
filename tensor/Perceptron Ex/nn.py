import numpy as np

x = np.array([1, 2])
W = np.array([[1, 2,3],[4,5,6]])

Y = np.dot(x, W)

print('shape of x = ', x.shape)
print(x)
print('shape of x = ', W.shape)
print(W)
print('shape of x = ', Y.shape)
print(Y)

np.exp()
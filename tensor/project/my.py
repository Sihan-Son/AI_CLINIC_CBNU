import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

xy_data = np.loadtxt("C:/Temp/dataTest.csv", delimiter=",", dtype=np.float32)

#전체 데이터
x_data = xy_data[:, 0: -1]
y_data = xy_data[:, [-1]]
#테스트 데이터
x_dataTest = x_data[980:]
y_dataTest = y_data[980:]

X = tf.placeholder(tf.float32, shape=[None, 30])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable((tf.random_normal([30, 1])), name="weight")
b = tf.Variable(tf.random_normal([1]), name='bias')

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=9e-6)
#optimizer = tf.train.AdamOptimizer(learning_rate=4)
train = optimizer.minimize(cost)

cost_list = []
acc_list = []
error_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        _c, _h, _ = sess.run([cost, H, train], feed_dict={X: x_data, Y: y_data})
        cost_list.append(_c)
    i = 0

    for data in x_dataTest:
        result = abs(sess.run(H, feed_dict={X:[data]}))
        real = y_dataTest[i]
        errorRate = abs(result - real) / result * 100
        if(result>=real):
            acc = real / result
        else:
            acc = result / real

        print("result = ", result, "real = ", real, "Accurancy = ", acc, "Error rate = ", errorRate)
        error_list.append(errorRate[0])
        acc_list.append(acc[0])
        i += 1

accAvg = np.mean(acc_list)
errorAvg = np.mean(error_list)
print("Cost : ", _c, ", Accurancy : ", accAvg, ", Error rate : ", errorAvg)

fig, ax = plt.subplots(1, 2)
fig.suptitle("Accurancy & Cost")
ax[0].plot(acc_list)
ax[1].plot(cost_list)
plt.show()
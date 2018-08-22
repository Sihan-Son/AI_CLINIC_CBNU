import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

xy_data = np.loadtxt("./data/fw.csv", delimiter=",", dtype=np.float32)

x_ = xy_data[:, 0: -1]*0.01
y_ = xy_data[:, [-1]]

test_data = x_[:100]
test_answer = y_[:100]

x_data = x_[100:]
y_data = y_[100:]

X = tf.placeholder(tf.float32, shape=[None, 30])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable((tf.random_normal([30, 1])), name="weight")
b = tf.Variable(tf.random_normal([1]), name='bias')

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

C_val =[]
acc_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _c, _h, _ = sess.run([cost, H, train], feed_dict={X: x_data, Y: y_data})
        C_val.append(_c)
        # if step % 100 == 0:
        #     print("step = ", step, "cost = ", _c, "\n")

    i=0
    for data in test_data:
        result = sess.run(H, feed_dict={X:[data]})
        real = test_answer[i]
        acc = 100 - ( abs(result - real) / result * 100 )
        print("result = ", result, "real = ", real, "Accurancy = ", acc)
        acc_list.append(acc[0])
        i += 1


plt.plot(C_val)
plt.show()
plt.plot(acc_list)
plt.ylim(50, 100)
plt.show()

print("단층 학습 최소 적중치 >> ", np.amin(acc_list))
print("단층 학습 평균 적중치 >> ",np.mean(acc_list))
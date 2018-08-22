import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X + b

cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

W_val = []
C_val = []
B_val = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _c, _w, _b, _ = sess.run([cost, W, b, train], feed_dict={X:x_data, Y:y_data})
        W_val.append(_w)
        C_val.append(_c)
        B_val.append(_b)
        if step%20 == 0:
            print(step, ":", _c, _w, _b)

    print('\nfinal W = ',_w, 'b= ', _b)
    test_data = [2,4,1,5,3]
    for data in test_data:
        print("X = ", data, "then Y =",sess.run(H,feed_dict={X:data}))

plt.plot(C_val)#, B_val, W_val)
plt.ylim(0, 0.00001)
plt.xlim(1000 ,2000)
plt.show()
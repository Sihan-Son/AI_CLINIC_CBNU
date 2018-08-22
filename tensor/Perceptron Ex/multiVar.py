import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [[73, 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]

y_data = [[152], [185], [180], [196], [142]]

# xy = np.loadtxt('data.txt',delimiter=",", dtype=np.float32)
# x_data = xy[:,0:-1]
# y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable((tf.random_normal([3, 1])), name="weight")
b = tf.Variable(tf.random_normal([1]))

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
cc = tf.Session().run(b)
C_val =[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _c, _h, _ = sess.run([cost, H, train], feed_dict={X:x_data, Y:y_data})
        C_val.append(_c)
        if step % 50 == 0:
            print("step = ", step, "cost = ", _c,"\n", _h)

    test_data = [[73, 80, 75],
                 [93, 88, 93],
                 [89, 91, 90]]
    for data in test_data:
        print("X = ", data, "then Y = ", sess.run(H, feed_dict={X:[data]}))


plt.plot(C_val)
plt.show()
print()
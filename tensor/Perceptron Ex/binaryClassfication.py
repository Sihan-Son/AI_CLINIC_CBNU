import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]

y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))
H = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(H) + (1 - Y) * tf.log(1 - H))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

predicted = tf.cast(H > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer()))

    for step in range(10001):
        _c, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            print("step = ", step, "cost = ", _c)

    print("\n Accuracy ")

    _h, _p, _a = sess.run([H, predicted, accuracy], feed_dict={X: x_data, Y: y_data})

    print("H >> ", _h)
    print("P >> ", _p)
    print("A >> ", _a)

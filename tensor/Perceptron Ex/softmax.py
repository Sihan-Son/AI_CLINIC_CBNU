import tensorflow as tf
import numpy as np


xy_data = np.array([[1, 2, 1, 1, 2],
                    [2, 1, 3, 2, 2],
                    [3, 1, 3, 4, 2],
                    [4, 1, 5, 5, 1],
                    [1, 7, 5, 5, 1],
                    [1, 2, 5, 6, 1],
                    [1, 6, 6, 6, 0],
                    [1, 7, 7, 7, 0]])

x_data = xy_data[:, 0:-1]
y_data = xy_data[:, [-1]]
nb_classes = 3

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([4, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost= tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(predicted, tf.argmax(Y_one_hot, 1))
accuarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step%100 == 0:
            loss, acc = sess.run([cost, accuarcy], feed_dict={X: x_data, Y: y_data})
            print("step:{:5}\tLoss:{:.3f}\tAcc:{:.2%}".format(step, loss, acc))

    print('\nTesting.....\n')
    pred = sess.run(predicted, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] prediction: {} True Y: {} ".format(p == int(y),p,int(y)))
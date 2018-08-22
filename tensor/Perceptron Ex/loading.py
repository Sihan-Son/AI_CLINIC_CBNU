from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
nb_classes = 10

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, nb_classes])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('./mnist_train_result.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    all_vars = tf.get_collection('vars')

    W1 = all_vars[0]
    b1 = all_vars[1]
    W2 = all_vars[2]
    b2 = all_vars[3]
    W3 = all_vars[4]
    b3 = all_vars[5]
    W4 = all_vars[6]
    b4 = all_vars[7]

    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

    hypothesis = tf.matmul(L3, W4) + b4

    is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


    print("Testing...............")
    print('Accuracy: ', accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))
    print()

    for i in range(3):
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("Predict: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r + 1], keep_prob: 1}))

        plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()
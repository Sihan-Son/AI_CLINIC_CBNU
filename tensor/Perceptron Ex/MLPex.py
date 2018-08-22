from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import time
from time import strftime

start_time = time.time()

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
nb_classes = 10

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.get_variable("W1", shape=[28*28, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.matmul(L3, W4) + b4

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)
cost = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

tf.add_to_collection('vars', W1)
tf.add_to_collection('vars', b1)
tf.add_to_collection('vars', W2)
tf.add_to_collection('vars', b2)
tf.add_to_collection('vars', W3)
tf.add_to_collection('vars', b3)
tf.add_to_collection('vars', W4)
tf.add_to_collection('vars', b4)
saver = tf.train.Saver()

training_epochs = 15
batch_size = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch
        saver.save(sess, './mnist_train_result')

        print('Epoch: %04d' % (epoch+1), 'cost = %.9f' % avg_cost)

    print("Testing...............")
    print('Accuracy: ', accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
    print()
print(time.time() - start_time)
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([28*28, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# train = optimizer.minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100
cost_val =[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        cost_val.append(avg_cost)
        print('Epoch: %04d' % (epoch+1), 'cost = %.9f' % avg_cost)

    print("Testing...............")
    print('Accuracy: ', accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    print()

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Predict: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r + 1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    plt.plot(cost_val)
    plt.show()

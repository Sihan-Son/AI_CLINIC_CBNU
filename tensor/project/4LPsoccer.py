import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
import random

start_time = time.time()

origin = np.loadtxt("./data/fw.csv", delimiter=",", dtype=np.float32)
data_norm = MinMaxScaler()
origin_data = data_norm.fit_transform(origin)
x_ = origin_data[:, 0: -1]
y_ = origin_data[:, [-1]]

test_data = x_[:100]
test_answer = y_[:100]

x_data = x_[100:]
y_data = y_[100:]

nb_classes = 1

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, 30])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.get_variable("W1", shape=[30, 30], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([30]), name='b1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[30, 30], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([30]), name='b2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[30, 30], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([30]), name='b3')
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[30, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_classes]), name='b4')

hypothesis = tf.matmul(L3, W4) + b4

# entropy = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)
entropy = tf.nn.relu(hypothesis)
cost = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-9).minimize(cost)

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

# training_epochs = 15
# batch_size = 9

cost_val = []
acc_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(r'./boardGraph', sess.graph)

    # for epoch in range(training_epochs):
    #     avg_cost = 0
    #     total_batch = int(test_answer / batch_size)

        # for i in range(total_batch):
    for step in range(10001):
        _c, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})

        cost_val.append(_c)
    saver.save(sess, './result_graph/fw')

    i = 0
    for data in test_data:
        # result = sess.run(hypothesis, feed_dict={X: [data], keep_prob: 1})
        # real = test_answer[i]
        # acc = 100 - ( abs(result - real) / result * 100 )
        # print("result = ", result, "real = ", real, "Accuracy = ", acc)

        result = abs(sess.run(hypothesis, feed_dict={X: [data], keep_prob: 1}))
        real = test_answer[i]
        errorRate = abs(result - real) / result  # * 100
        if result >= real:
            acc = real / result
        else:
            acc = result / real

        print("result = ", result, "real = ", real, "Accuracy = ", acc, "Error rate = ", errorRate)

        acc_list.append(acc[0])
        i += 1

    acc2 = accuracy.eval(session=sess, feed_dict={X: test_data, Y: test_answer, keep_prob: 1})
    print('Accuracy: ', acc2)
    acc_list.append(acc2)

    for i in range(3):
        #r = random.randint(0, len(test_answer) - 1)
        print("Label: ", test_answer[i])
        print("Predict: ", sess.run(hypothesis, feed_dict={X: [test_data[i]], keep_prob: 1}))




plt.plot(cost_val)
plt.suptitle("cost")
plt.savefig("cost4L.png")

plt.show()
plt.plot(acc_list)
plt.suptitle("acc")
plt.savefig("acc4L.png")
plt.show()

print("단층 학습 최소 적중치 >> ", np.amin(acc_list))
print("단층 학습 최소 적중치 >> ", np.amax(acc_list))
print("단층 학습 평균 적중치 >> ", np.mean(acc_list))
# print(test_answer)
print(time.time() - start_time)

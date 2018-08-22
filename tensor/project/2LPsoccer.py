import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
import random

### 데이터 불러오기
origin_data = np.loadtxt("./data/fw.csv", delimiter=",", dtype=np.float32)
### 정규화
data_norm = MinMaxScaler()
origin_data = data_norm.fit_transform(origin)

### 데이타 가공
x_ = origin_data[:, 0: -1]
y_ = origin_data[:, [-1]]

test_data = x_[:100]
test_answer = y_[:100]

x_data = x_[100:]
y_data = y_[100:]

### 최종 도출 값 개수
nb_classes = 1

### 변수 정의
keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, shape=[None, 30])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.get_variable("W1", shape=[30, 15], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([15]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[15, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.matmul(L1, W2) + b2


# for l in range(641, 662, 10):
Relu = tf.nn.relu(hypothesis)
# entropy = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)
cost = tf.reduce_mean(Relu)
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

cost_val = []
acc_list = []
error_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(r'./boardGraph', sess.graph)
    for step in range(10001):
        _c, _r, _ = sess.run([cost, Relu, optimizer], feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
        cost_val.append(_c)

        # if step % 1000 == 0:
        #     print(_c)

    i = 0
    # wirteData = str(l)+"\n "
    for data in test_data:

        result = abs(sess.run(hypothesis, feed_dict={X: [data], keep_prob: 1}))
        real = test_answer[i]
        errorRate = abs(result - real) / (result * 100)

        if result >= real:
            acc = real / result
        else:
            acc = result / real

        if i%20 == 0:
            tex =str(("result = " + str(result[0][0])+ " real = "+ str(real)+ "  Accuracy = "+ str(acc)))
            "result = ", result[0][0], "real = ", real, "Accuracy = ", acc
            print(tex)#, "Error rate = ", errorRate)
            print(acc.shape)
            acc = accuracy.eval(session=sess, feed_dict={X: test_data, Y: test_answer, keep_prob: 1})
            print('Accuracy: ', acc)

        #error_list.append(errorRate[0])
        acc_list.append(acc[0][0])
        i += 1
    #     wirteData +="\n"+str(tex)
    # if np.amin(acc_list) >= 0.73:
    #     with open("test.txt", "a") as f:
    #         f.write(wirteData)

plt.plot(cost_val)
plt.suptitle("cost")
plt.savefig("cost.png")
plt.show()
# tt = "./acc"
# tt += str(l)
# tt += ".png"
plt.plot(acc_list)
plt.suptitle("acc >>")
# plt.savefig(tt)
plt.show()
# #
print("이층 학습 최소 적중치 >> ", np.amin(acc_list))
print("이층 학습 최고 적중치 >> ", np.amax(acc_list))
print("이층 학습 평균 적중치 >> ", np.mean(acc_list))
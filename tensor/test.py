import os
import tensorflow as tf
import matplotlib .pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



"""세션 사용방법"""
#
# sum_V = tf.Variable(0)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for i in range(5):
#     sum_V = sum_V + 1
#     print(sess.run(sum_V))
# sess.close()


"""feed_dict"""
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
#
# output = tf.multiply(input1, input2)
#
# with tf.Session() as sess:
#     print(sess.run(output, feed_dict={input1:7.0, input2:2.0}))


"""모델 만들기"""
# apple = [2, 3, 4, 5]
# oranges = [3, 5, 7, 9]
#
#
# price_of_apple = tf.constant(100, dtype=tf.int16)
# price_of_orange = tf.constant(150, dtype=tf.int16)
# tax = tf.constant(1.1, dtype=tf.float32)
#
# num_of_apple = tf.placeholder(dtype=tf.int16)
# num_of_orange = tf.placeholder(dtype=tf.int16)
#
# price_of_apple = tf.multiply(price_of_apple, num_of_apple)
# price_of_orange = tf.multiply(price_of_orange, num_of_orange)
#
# total = tf.add(price_of_orange, price_of_apple)
#
# totalCost = tf.multiply(tf.cast(total, tf.float32), tax)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(4):
#         _c = sess.run(totalCost, feed_dict={num_of_apple: apple[i], num_of_orange: oranges[i]})
#
#         print("apple = ", apple[i], "orange = ", oranges[i], "then total price = ", _c)


'''tensor board graphEx'''
# num1 = tf.constant(3.0)
# num2 = tf.constant(2.0)
# num3 = tf.constant(5.0)
#
# intermed = tf.add(num1, num2)
#
# mul = tf.multiply(num1, intermed)
#
# with tf.Session() as sess:
#     print(sess.run([mul, intermed]))
#
#     summary = tf.summary.merge_all()
#     writer = tf.summary.FileWriter(r'D:\Project\logs', sess.graph)
#     # writer.add_graph(sess.graph)


a = tf.constant(3.0, name='a')
b = tf.constant(5.0, name='b')
c = a * b

c_summary = tf.summary.scalar("point", c)
cc = tf.summary.histogram("hio",tf.add(a,b))
summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(r"./logs", sess.graph)

    result = sess.run([summary])
    writer.add_summary(result[0])

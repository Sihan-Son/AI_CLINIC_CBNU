import tensorflow as tf
import matplotlib.pyplot as plt

X = tf.constant([1,2,3,4,5],dtype=tf.float32)
Y = tf.constant([1,2,3,4,5],dtype=tf.float32)

W = tf.placeholder(tf.float32)
hypothesis = W * X
cost = tf.reduce_mean(tf.square(hypothesis - Y))

W_val = []
cost_val = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(-30,50):
        w, c = sess.run([W, cost], feed_dict={W:i*0.1})
        if i%10 == 0:
            print("w = ", w, end="\t")
            print("c = ", c)
        W_val.append(w)
        cost_val.append(c)

plt.plot(W_val, cost_val)
plt.show()
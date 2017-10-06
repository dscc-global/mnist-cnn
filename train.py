#!/usr/bin/env python3
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import csv

csv_file = open('data.csv', 'w')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

# Initial functions for future use with noise
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return initial

# Initial functions for defining layers including CNN layer and Pooling layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Input placeholder & converting 768d vector to 28x28 matrix
x  = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10 ], name='y_')
x_img = tf.reshape(x, [-1, 28, 28, 1], name='x_img')

# Defining CNN Layer 1 with Pooling and ReLU
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1, name='h_conv1')
h_pool1 = max_pool_2x2(h_conv1)

# Defining CNN Layer 2 with Pooling and ReLU
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')
h_pool2 = max_pool_2x2(h_conv2)

# Reshaping & FCN
W_fc1 = weight_variable([49*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 49*64], name='h_pool2_flat')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')

# Dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

# Softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Loss function & Optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Evaluating
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
saver = tf.train.Saver()
tf.global_variables_initializer().run()
# Train for 20000 steps & saving data to a csv file
try:
    csv_handler = csv.writer(csv_file)
    csv_handler.writerow(('step', 'accuracy'))
    print('step, accuracy')
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('{0}, {1}'.format(i, train_accuracy))
        csv_handler.writerow((i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})
finally:
    csv_file.close()
# Save & test
saver.save(sess, 'mnist_cnn')
print('accuracy: {0}'.format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))

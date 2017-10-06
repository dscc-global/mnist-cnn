#!/usr/bin/env python3
import argparse
import sys
import time
import PIL
import tensorflow as tf

start = 0
end = 0

class ImgError(Exception):
    pass

def prep(mat):
    # Denoising
    for i, val in enumerate(mat):
        if val < .1:
            mat[i] = 0

def load(filename):
    gray = PIL.Image.open(filename).convert('L')
    width, height = gray.size
    if not width == height == 28:
        raise ImgError
    else:
        mat = list(gray.getdata())
        mat_std = [float((255-a)/255) for a in mat]
        prep(mat_std)
        return mat_std

def predict(matrix):
    saver = tf.train.import_meta_graph('mnist_cnn.meta')
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        y = graph.get_tensor_by_name('y:0')
        prediction = tf.argmax(y, 1)
        start = time.time()
        result = prediction.eval(feed_dict={x: [matrix], keep_prob: 1.0})[0]
        end = time.time()
        return result

if __name__ == '__main__':
    try:
        print('Result: {0}'.format(predict(load(sys.argv[1]))))
        print('Time: {0}s'.format(end - start))
    except ImgError:
        print('Please resize your image to 28x28.')
    except IndexError:
        print('Usage: python3 predict.py image.jpg')

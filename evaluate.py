#!/usr/bin/env python3
import sys
import time
import PIL
import tensorflow as tf
import os

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

def evaluate(filenames):
    correct = 0
    all = 0
    start = time.time()
    for file in filenames:
        result = prediction.eval({x: [load(file)], keep_prob: 1.0})
        print(file, result, result == int(file.split('/')[1][:1]))
        if result == int(file.split('/')[1][:1]):
            correct += 1
        all += 1
    end = time.time()
    print('accuracy: {}%, avg time: {} s'.format(correct / all * 100, (end - start) / len(filenames)))

if __name__ == '__main__':
    files = os.listdir('test_data/')
    for i, file in enumerate(files):
        if file[:1] == '.':
            del files[i]
    for i, file in enumerate(files):
        files[i] = 'test_data/' + files[i]
    saver = tf.train.import_meta_graph('mnist_cnn.meta')
    sess = tf.InteractiveSession()
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    y = graph.get_tensor_by_name('y:0')
    prediction = tf.argmax(y, 1)
    evaluate(files)


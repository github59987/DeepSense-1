#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: ceshi
@time: 2019/3/3 11:13
@desc:
'''
import tensorflow as tf
import numpy as np
from tensorflow_own.Routine_operation import SaveFile, LoadFile








if __name__ == '__main__':
    # a = tf.constant(np.arange(20).reshape(2, 2, 5), dtype=tf.float32)
    # b = tf.constant(np.array([[[1]]]), dtype= tf.float32)
    # c = tf.concat([a, b], axis=1)
    # with tf.Session() as sess:
    #     a_ = sess.run(a)
    #     b_ = sess.run(b)
    #     c_ = sess.run(c)
    #     print(c_.shape)
    a = np.arange(20).reshape(4, 5)
    b = np.array([1])
    # c = np.concatenate((a, b), axis=0)
    print(a.shape, b.shape)
    print(a + b)

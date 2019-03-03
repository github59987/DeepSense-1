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

a = tf.constant(np.arange(4).reshape(1, 1, 4, 1), dtype= tf.float32)
b = tf.nn.conv2d(input= a,
                 filter= tf.Variable(tf.truncated_normal(shape= (1, 2, 1, 1), mean= 0, stddev= 1), dtype= tf.float32),
                 strides= [1, 1, 3, 1],
                 padding= 'VALID')
print(b)
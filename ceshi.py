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
from tensorflow_own.TFrecord_operation import FileoOperation







if __name__ == '__main__':
    # def fun(a):
    #     if a:
    #         return 1
    #     else:
    #         return 2
    #
    a = tf.placeholder(dtype=tf.bool)
    def fun(num):
        return tf.constant(num, dtype= tf.float32)
    def fun2(a):
        b = tf.cond(a, lambda:fun(1), lambda:fun(2))
        return b
    # numm = fun(a)
    with tf.Session() as sess:
        num = sess.run(fun2(a), feed_dict={a: True})
        print(num)


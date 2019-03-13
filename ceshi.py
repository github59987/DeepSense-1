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
    a = tf.constant(np.arange(20).reshape(4, 5), dtype=tf.float32)
    print(np.array(a.get_shape().as_list()) + 3)

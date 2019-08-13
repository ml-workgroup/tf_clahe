#!/usr/bin/env python

import tensorflow as tf
import numpy as np
module = tf.load_op_library('bin/clahe_op_kernel.so')

window_size = 3
relaive_clip_limit = 0.01
multiplicative_redistribution = True
with tf.Session('') as sess:
        input_array = np.array([[[1, 2], [3, 4]]], dtype=np.float)
        print(input_array.shape)

        ret = module.clahe(input_array, window_size, relaive_clip_limit, multiplicative_redistribution)
        ret = ret.eval()
print(ret)
exit(0)

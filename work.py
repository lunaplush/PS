# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:27:14 2017

@author: Inspiron
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

#%%

#df = pd.DataFrame({'X':np.random.standard_normal(100), 'Y':np.random.standard_cauchy(100)}, index = np.linspace(1,50,100))
#df.plot('X','Y', kind = 'scatter')
#df.plot()

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
output = tf.multiply(x,y)

with tf.Session() as sess:
    result = sess.run(output, feed_dict={x:3, y:10})
print(result)    

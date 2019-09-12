#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten , Dropout, Dense
from input_data import image_attributes, gen_yolo_data
import numpy as np


# In[2]:


# getting data
image_attributes


# In[3]:


images, vectors = gen_yolo_data()


# In[4]:


images = [np.divide(img,255,dtype='float32') for img in images]


# In[5]:


images[0].dtype


# In[6]:


# input data dimensions with batch
X_dims = (None,) + image_attributes['img_res'] + (1,)
Y_dims = (None,) + vectors[0].shape


# In[ ]:





# In[7]:


# model
X = tf.placeholder(tf.float32,shape=X_dims)
Y = tf.placeholder(tf.float32,shape=Y_dims)


# In[ ]:





# In[8]:


conv1 = Conv2D(10,5)(X)


# In[9]:


last = Flatten()(conv1)
f = tf.reduce_sum(last)


# In[10]:



# In[11]:


initop = tf.global_variables_initializer()



# In[ ]:


with tf.Session() as sess:
    sess.run(initop)
    out = sess.run(f,feed_dict={X:images[:2],Y:vectors[:2]})
    print(out)

# In[ ]:





import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np
import os
from keras.utils import np_utils
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import cv2
import time 
from scipy.ndimage.filters import gaussian_filter
import random
import atexit
import math


'''

Convolves an image, either leaving the scale the same or scaling down

'''
def ImageDownConvBlock(inp_layer, out_scale = 2):
	conv_1 = tf.keras.layers.Conv2D(3, 3, padding = 'same', strides = 1, activation = 'relu')(inp_layer)
	conv_2 = tf.keras.layers.Conv2D(3, 3, padding = 'same', strides = 1, activation = 'relu')(conv_1)
	conv_3 = tf.keras.layers.Conv2D(3, 3, padding = 'same', strides = 1, activation = 'relu')(conv_2)

	# This is where we downsample
	conv_final = tf.keras.layers.Conv2D(3, out_scale, padding = 'same', strides = out_scale, activation = 'relu')(conv_3)

	return conv_final


'''

Convolves an image, ether leaving the scale the same or scaling up

'''
def ImageUpConvBlock(inp_layer, out_scale = 2):
	conv_1 = tf.keras.layers.Conv2DTranspose(3, 3, padding = 'same', strides = 1, activation = 'relu')(inp_layer)
	conv_2 = tf.keras.layers.Conv2DTranspose(3, 3, padding = 'same', strides = 1, activation = 'relu')(conv_1)
	conv_3 = tf.keras.layers.Conv2DTranspose(3, 3, padding = 'same', strides = 1, activation = 'relu')(conv_2)

	# This is where we downsample
	conv_final = tf.keras.layers.Conv2DTranspose(3, out_scale, padding = 'same', strides = out_scale, activation = 'relu')(conv_3)

	return conv_final


'''

Convolves an image an arbitrary number of times while retaining scale

'''
def ImageConvBlock(inp_layer, n = 3):
	ret = inp_layer
	for i in range(n):
		ret = tf.keras.layers.Conv2D(3, 3, padding = 'same', strides = 1, activation = 'relu')(ret)
	return ret


'''

Creates a block to process time input given a vector

'''
def TimeProcBlock(n_time_steps, out_shape):
	n_outputs = np.prod(out_shape)

	inp_t = tf.keras.layers.Input(shape = n_time_steps)
	den = tf.keras.layers.Dense(n_outputs)(inp_t)
	res = tf.keras.layers.Reshape(out_shape)(den)

	return inp_t, res


'''

Downscales a convolved time embedding

'''
def DownscaledTime(time_tensor, scale = 2):
        return tf.keras.layers.Conv2D(3, 3, strides = scale, padding = 'same', activation = 'relu')(time_tensor)


'''

Creates and returns a network model

'''
def network_model(im_sidelen, n_time_steps):
	inp_i = tf.keras.layers.Input(shape = (im_sidelen, im_sidelen, 3))
	inp_t, conv_t = TimeProcBlock(n_time_steps, (im_sidelen, im_sidelen, 3))

	conv_t_050 = DownscaledTime(conv_t)
	conv_t_025 = DownscaledTime(conv_t_050)

	added_im_t = tf.keras.layers.Add()([inp_i, conv_t])

	d1 = ImageDownConvBlock(added_im_t)
	added_d1_t = tf.keras.layers.Add()([d1, conv_t_050])
	d2 = ImageDownConvBlock(added_d1_t)
	added_d2_t = tf.keras.layers.Add()([d2, conv_t_025])
	d3 = ImageDownConvBlock(added_d2_t)

	latent_conv = ImageConvBlock(d3, 5)

	u3 = ImageUpConvBlock(latent_conv)
	added_u3_t = tf.keras.layers.Add()([u3, conv_t_025])
	u2 = ImageUpConvBlock(added_u3_t)
	added_u2_t = tf.keras.layers.Add()([u2, conv_t_050])
	u1 = ImageUpConvBlock(added_u2_t)

	model = tf.keras.Model(inputs = [inp_i, inp_t], outputs = u1)

	model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(
		learning_rate = 1e-4
		)
	)

	return model



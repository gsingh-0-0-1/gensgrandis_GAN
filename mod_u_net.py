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
def ImageDownConvBlock(inp_layer, out_scale = 2, n = 5):
        ret = inp_layer
        for i in range(n):
                f = 12
                if i == n - 1:
                        f = 3
                ret = tf.keras.layers.Conv2D(f, 3, padding = 'same', strides = 1, activation = 'relu')(ret)

        # This is where we downsample
        conv_final = tf.keras.layers.Conv2D(3, out_scale, padding = 'same', strides = out_scale, activation = 'relu')(ret)

        return ret, conv_final


'''

Convolves an image, ether leaving the scale the same or scaling up

'''
def ImageUpConvBlock(inp_layer, out_scale = 2, n = 5):
        ret = inp_layer
        for i in range(n):
                ret = tf.keras.layers.Conv2DTranspose(12, 3, padding = 'same', strides = 1, activation = 'relu')(ret)

        # This is where we upsample
        conv_final = tf.keras.layers.Conv2DTranspose(3, out_scale, padding = 'same', strides = out_scale, activation = 'relu')(ret)

        return conv_final


'''

Convolves an image an arbitrary number of times while retaining scale

'''
def ImageConvBlock(inp_layer, n = 3):
	ret = inp_layer
	for i in range(n - 1):
		ret = tf.keras.layers.Conv2D(128, 3, padding = 'same', strides = 1, activation = 'relu')(ret)
	ret = tf.keras.layers.Conv2D(3, 3, padding = 'same', strides = 1, activation = 'linear')(ret)
	return ret


'''

Creates a block to process time input given a vector

'''
def TimeProcBlock(n_time_steps, out_shape):
	n_outputs = np.prod(out_shape)

	inp_t = tf.keras.layers.Input(shape = n_time_steps)
	den = tf.keras.layers.Dense(n_outputs, activation = 'relu')(inp_t)
	den = tf.keras.layers.Dense(n_outputs)(den)
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

        added_im_t = tf.keras.layers.Concatenate(axis = -1)([inp_i, conv_t])

        #initial_conv = ImageConvBlock(added_im_t, 5)

        penult_d1, d1 = ImageDownConvBlock(added_im_t, n = 8)
        added_d1_t = tf.keras.layers.Concatenate(axis = -1)([d1, conv_t_050])
        penult_d2, d2 = ImageDownConvBlock(added_d1_t, n = 3)

        latent_conv = ImageConvBlock(d2, 5)

        u2 = ImageUpConvBlock(latent_conv, n = 3)
        added_u2_t = tf.keras.layers.Concatenate(axis = -1)([u2, conv_t_050, penult_d2])
        u1 = ImageUpConvBlock(added_u2_t, n = 3)

        final_add = tf.keras.layers.Concatenate(axis = -1)([u1, conv_t, penult_d1])

        final_conv = ImageConvBlock(final_add, 5)

        model = tf.keras.Model(inputs = [inp_i, inp_t], outputs = final_conv)

        model.compile(loss = 'mae', optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate = 1e-3
                )
        )

        return model



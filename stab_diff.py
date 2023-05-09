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

INIT_LR = 1e-4
NOISE_DIM = 300
TARGET_IMAGE_SIDELENGTH = 20

def load_image_dataset():

	TARGET_IMAGE_SIDELENGTH = 20

	assert TARGET_IMAGE_SIDELENGTH % 2 == 0

	train_images = []
	for fname in os.listdir("./train_data/"):
		if str(fname).startswith('map'):
			img = cv2.cvtColor(cv2.imread("./train_data/" + fname), cv2.COLOR_BGR2RGB)
			train_images.append(img)

	final = []

	for img in train_images[:30]:
		for i in range(int(1000 / TARGET_IMAGE_SIDELENGTH)):
			for j in range(int(1000 / TARGET_IMAGE_SIDELENGTH)):
				final.append(img[TARGET_IMAGE_SIDELENGTH * i : TARGET_IMAGE_SIDELENGTH * (i + 1), TARGET_IMAGE_SIDELENGTH * j : TARGET_IMAGE_SIDELENGTH * (j + 1)])

	train_images = final

	train_images = np.array(train_images)

	#scale data from 0 to 1
	train_images = (train_images / 255)

	print(np.amin(train_images), np.amax(train_images))

	#train_images = train_images

	#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

	return train_images


def add_noise_to_dataset(dataset, steps = 10):
	noise = np.random.random(dataset.shape)

	final = []

	noisy = dataset

	for i in range(steps + 1):
		noisy = (noise * i / steps) + (dataset * (steps - i) / steps)

		final.append(noisy)

	return np.array(final[::-1])

N_STEPS = 100

train_dataset = load_image_dataset()
#train_dataset = add_noise_to_dataset(train_dataset, N_STEPS)

N_EXAMPLES = train_dataset.shape[0]


print("%d training examples with %d noise steps" % (N_EXAMPLES, N_STEPS))

PIXELS_PER_IMG = 3 * TARGET_IMAGE_SIDELENGTH * TARGET_IMAGE_SIDELENGTH

DOWNSAMPLED_SIDELEN = int(TARGET_IMAGE_SIDELENGTH / 2)


class TotalSquaredError(tf.keras.losses.Loss):

	def call(self, y_true, y_pred):
		return tf.reduce_sum(tf.math.square(y_pred - y_true))

inp_i = tf.keras.layers.Input(shape = (PIXELS_PER_IMG))
inp_t = tf.keras.layers.Input(shape = (PIXELS_PER_IMG))

#-- First Block --#
add_1_1 = tf.keras.layers.Concatenate()([inp_i, inp_t])

den_1_1 = tf.keras.layers.Dense(PIXELS_PER_IMG, activation = 'relu')(add_1_1)
den_1_2 = tf.keras.layers.Dense(PIXELS_PER_IMG / 4, activation = 'relu')(den_1_1)

reshape = tf.keras.layers.Reshape((DOWNSAMPLED_SIDELEN, DOWNSAMPLED_SIDELEN, 3))(den_1_2)

con_1_1 = tf.keras.layers.Conv2D(3, 3, activation = 'relu')(reshape)
con_1_2 = tf.keras.layers.Conv2D(3, 3, activation = 'relu')(con_1_1)
con_1_3 = tf.keras.layers.Conv2D(3, 3, activation = 'relu')(con_1_2)
con_1_4 = tf.keras.layers.Conv2D(3, 3, activation = 'relu')(con_1_3)

dec_1_1 = tf.keras.layers.Conv2DTranspose(3, 3, activation = 'relu')(con_1_4)
dec_1_2 = tf.keras.layers.Conv2DTranspose(3, 3, activation = 'relu')(dec_1_1)
dec_1_3 = tf.keras.layers.Conv2DTranspose(3, 3, activation = 'relu')(dec_1_2)
dec_1_4 = tf.keras.layers.Conv2DTranspose(3, 3, activation = 'relu')(dec_1_3)

flatten = tf.keras.layers.Flatten()(dec_1_4)

den_1_3 = tf.keras.layers.Dense(PIXELS_PER_IMG / 4, activation = 'relu')(flatten)
den_1_4 = tf.keras.layers.Dense(PIXELS_PER_IMG, activation = 'relu')(den_1_3)

add_1_2 = tf.keras.layers.Concatenate()([den_1_4, inp_t])

den_1_5 = tf.keras.layers.Dense(PIXELS_PER_IMG, activation = 'relu')(add_1_2)

model = tf.keras.Model(inputs=[inp_i, inp_t], outputs = den_1_5)

model.compile(optimizer = 'adam', loss = 'mse')


flat_shape = (N_EXAMPLES, PIXELS_PER_IMG)


noise = np.random.random(train_dataset.shape)

'''
for i in range(N_STEPS + 1):
	new = (train_dataset[0] * i / N_STEPS) + (noise[0] * (N_STEPS - i) / N_STEPS)
	plt.imshow(new)
	plt.pause(0.01)
'''

atexit.register(model.save, "saved_model")
for jump in range(N_STEPS):
	print("TRAINING FOR JUMP %d to %d" % (jump, jump + 1))

	one = (train_dataset * jump / N_STEPS) + (noise * (N_STEPS - jump) / N_STEPS)
	two = (train_dataset * (jump + 1) / N_STEPS) + (noise * (N_STEPS - jump - 1) / N_STEPS)

	flat_1 = one.reshape(flat_shape)
	flat_2 = two.reshape(flat_shape)

	model.fit([flat_1, np.full_like(flat_1, jump)], flat_2, 
		batch_size = 10,
		epochs = 25)

noise = np.array(np.random.random((4, PIXELS_PER_IMG)))
#noise = noise + STEPS - 1

for i in range(N_STEPS):
	out = model.predict(np.concatenate((noise, np.full_like(noise, i)), axis = -1))
	plt.subplot(2, 2, 1)
	plt.imshow(out[0].reshape((TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)) / np.amax(out))
	plt.subplot(2, 2, 2)
	plt.imshow(out[1].reshape((TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)) / np.amax(out))
	plt.subplot(2, 2, 3)
	plt.imshow(out[2].reshape((TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)) / np.amax(out))
	plt.subplot(2, 2, 4)
	plt.imshow(out[3].reshape((TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)) / np.amax(out))
	plt.show()
	noise = out

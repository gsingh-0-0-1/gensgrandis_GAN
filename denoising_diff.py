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

print(tf.__version__)

print("Devices: ", [el for el in tf.config.list_physical_devices()])

INIT_LR = 1e-4
NOISE_DIM = 300
TARGET_IMAGE_SIDELENGTH = 20

def load_image_dataset():

  TARGET_IMAGE_SIDELENGTH = 20

  assert TARGET_IMAGE_SIDELENGTH % 2 == 0

  train_images = []
  for fname in os.listdir("train_data/"):
    if str(fname).startswith('map'):
      img = cv2.cvtColor(cv2.imread("train_data/" + fname), cv2.COLOR_BGR2RGB)
      train_images.append(img)

  final = []

  for img in train_images[:10]:
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






N_STEPS = 20

train_dataset = load_image_dataset()
#train_dataset = add_noise_to_dataset(train_dataset, N_STEPS)

N_EXAMPLES = train_dataset.shape[0]


print("%d training examples with %d noise steps" % (N_EXAMPLES, N_STEPS))

PIXELS_PER_IMG = 3 * TARGET_IMAGE_SIDELENGTH * TARGET_IMAGE_SIDELENGTH

DOWNSAMPLED_SIDELEN = int(TARGET_IMAGE_SIDELENGTH / 2)

SLEN = TARGET_IMAGE_SIDELENGTH


inp_i = tf.keras.layers.Input(shape = (PIXELS_PER_IMG))

reshape = tf.keras.layers.Reshape((SLEN, SLEN, 3))(inp_i)

res_1_conv_1 = tf.keras.layers.Conv2D(128, 3, strides = 1, padding = 'same')(reshape)
res_1_conv_2 = tf.keras.layers.Conv2D(128, 3, strides = 1, padding = 'same')(res_1_conv_1)
res_1_conv_3 = tf.keras.layers.Conv2D(128, 3, strides = 1, padding = 'same')(res_1_conv_2)

res_05_conv_1 = tf.keras.layers.Conv2D(128, 2, strides = 2)(res_1_conv_3)

res_025_conv_1 = tf.keras.layers.Conv2D(256, 2, strides = 2)(res_05_conv_1)
res_025_conv_2 = tf.keras.layers.Conv2D(256, 3, strides = 1, padding = 'same')(res_025_conv_1)
res_025_conv_3 = tf.keras.layers.Conv2D(512, 3, strides = 1, padding = 'same')(res_025_conv_2)
res_025_conv_4 = tf.keras.layers.Conv2D(512, 3, strides = 1, padding = 'same')(res_025_conv_3)
res_025_conv_5 = tf.keras.layers.Conv2D(256, 3, strides = 1, padding = 'same')(res_025_conv_4)
res_025_conv_6 = tf.keras.layers.Conv2D(256, 3, strides = 1, padding = 'same')(res_025_conv_5)

upsamp_025_05 = tf.keras.layers.UpSampling2D(size = (2, 2))(res_025_conv_6)
concat_res_05 = tf.keras.layers.Concatenate(axis = -1)([upsamp_025_05, res_05_conv_1])
res_05_conv_2 = tf.keras.layers.Conv2D(256, 3, strides = 1, padding = 'same')(concat_res_05)
res_05_conv_3 = tf.keras.layers.Conv2D(128, 3, strides = 1, padding = 'same')(res_05_conv_2)

upsamp_05_1 = tf.keras.layers.UpSampling2D(size = (2, 2))(res_05_conv_3)
concat_res_1 = tf.keras.layers.Concatenate(axis = -1)([upsamp_05_1, res_1_conv_3])
res_1_conv_3 = tf.keras.layers.Conv2D(128, 3, strides = 1, padding = 'same')(concat_res_1)
res_1_conv_4 = tf.keras.layers.Conv2D(3, 3, strides = 1, padding = 'same')(res_1_conv_3)

out = tf.keras.layers.Flatten()(res_1_conv_4)

model = tf.keras.Model(inputs = inp_i, outputs = out)


model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(
    learning_rate = 1e-5
))


def noise_add_frac(frac):
  return frac
  #return 1 - ( (1 - frac ** 2) ** (1 / 2))

atexit.register(model.save, "/content/drive/My Drive/GG_train_data/saved_model")

selected_train_dataset = train_dataset[:5000]
noisy = selected_train_dataset
one = noisy

flat_shape = (selected_train_dataset.shape[0], PIXELS_PER_IMG)

images = []
noises = []

noise = np.random.random(size = selected_train_dataset.shape)
add_noise = (noise - one) * noise_add_frac(1 / N_STEPS)

for jump in range(N_STEPS):
  two = one + add_noise

  flat_2 = two.reshape(flat_shape)
  
  images.append(flat_2)
  noises.append(add_noise.reshape(flat_shape))

  one = two

images = np.concatenate(tuple(images), axis = 0)
noises = np.concatenate(tuple(noises), axis = 0)

print(images.shape, noises.shape)

n_total = images.shape[0]

training_frac = 0.7

splitpoint = int(n_total * training_frac)

tf.device('/device:GPU:0')

model.fit(images[:splitpoint], noises[:splitpoint], validation_data = (images[splitpoint:], noises[splitpoint:]),
      batch_size = 5,
      epochs = 10,
      shuffle = True)

def generate_test_images():
  noise = np.random.random(size = (4, TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3))
  subtracted = noise
  for i in range(N_STEPS * 2):
    plt.clf()
    plt.suptitle("Step %d of %d" % (i + 1, N_STEPS))
    pred_noise = model.predict(subtracted.reshape(4, PIXELS_PER_IMG))
    subtracted = subtracted - pred_noise.reshape(subtracted.shape) 
    plt.subplot(2, 2, 1)
    plt.imshow(subtracted[0])    
    plt.subplot(2, 2, 2)
    plt.imshow(subtracted[1])    
    plt.subplot(2, 2, 3)
    plt.imshow(subtracted[2])   
    plt.subplot(2, 2, 4)
    plt.imshow(subtracted[3])    
    if i + 1 == N_STEPS or i + 1 == 2 * N_STEPS:
      plt.show()
    else:
      plt.pause(0.1)

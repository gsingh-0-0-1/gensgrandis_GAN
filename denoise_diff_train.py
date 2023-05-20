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
import keras_nlp

from keras.layers import Activation
from keras import backend as K

import mod_u_net

print(tf.__version__)

#print("Devices: ", [el for el in tf.config.list_physical_devices()])

INIT_LR = 1e-4
NOISE_DIM = 300
TARGET_IMAGE_SIDELENGTH = 16

def load_image_dataset():

  TARGET_IMAGE_SIDELENGTH = 16

  assert TARGET_IMAGE_SIDELENGTH % 2 == 0

  train_images = []
  for fname in os.listdir("train_data/"):
    if str(fname).startswith('map'):
      img = cv2.cvtColor(cv2.imread("train_data/" + fname), cv2.COLOR_BGR2RGB)
      train_images.append(img)

  final = []

  for img in train_images[:5]:
    for i in range(int(1000 / TARGET_IMAGE_SIDELENGTH)):
      for j in range(int(1000 / TARGET_IMAGE_SIDELENGTH)):
        new = img[TARGET_IMAGE_SIDELENGTH * i : TARGET_IMAGE_SIDELENGTH * (i + 1), TARGET_IMAGE_SIDELENGTH * j : TARGET_IMAGE_SIDELENGTH * (j + 1)]
        #we don't want monochrome (all-green, in this case) images
        if new[:, :, 1].std() < 10:
          continue
        final.append(new)

  train_images = final

  train_images = np.array(train_images)

  #scale data from 0 to 1
  train_images = (train_images / 255)

  print(np.amin(train_images), np.amax(train_images))

  #train_images = train_images

  #train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

  return train_images






N_STEPS = 3

train_dataset = load_image_dataset()
#train_dataset = add_noise_to_dataset(train_dataset, N_STEPS)

N_EXAMPLES = train_dataset.shape[0]


print("%d training examples with %d noise steps" % (N_EXAMPLES, N_STEPS))

PIXELS_PER_IMG = 3 * TARGET_IMAGE_SIDELENGTH * TARGET_IMAGE_SIDELENGTH

DOWNSAMPLED_SIDELEN = int(TARGET_IMAGE_SIDELENGTH / 2)

SLEN = TARGET_IMAGE_SIDELENGTH

TIME_MATRICES = []

for i in range(N_STEPS):
  arr = np.zeros((N_STEPS))
  #for idx, el in np.ndenumerate(arr):
  #  arr[idx] = np.exp(-((i - idx[0]) ** 2 + (i - idx[1]) ** 2))
  arr[i] = 1
  TIME_MATRICES.append(arr)

def gen_time_matrix(t, total):
  return TIME_MATRICES[t]

def time_conv_block(inp, out_slen):
  t_flat = tf.keras.layers.Flatten()(inp)
  t_dense = tf.keras.layers.Dense(out_slen ** 2)(t_flat)
  t_dense_2 = tf.keras.layers.Dense(out_slen ** 2, activation = 'sigmoid')(t_dense)
  t_reshape = tf.keras.layers.Reshape((out_slen, out_slen, 1))(t_dense)
  return t_reshape

def im_time_conc(im, t, out_slen):
  return tf.keras.layers.Concatenate(axis = -1)([im, time_conv_block(t, out_slen)])

def custom_sigmoid(x):
  return (K.sigmoid(x) - 0.5) * (2 / N_STEPS)


#p = 1 is linear
#p = 2 creates a quarter-circle pattern
def power_pattern_noise(i, n, p = 1):
  frac = i / n
  return 1 - ( (1 - (frac ** p)) ** (1 / p))

def sqrt_noise(i, n):
  return (i / n) ** (1 / 2)

def noise_add_frac(i, n):
  p = 1
  return power_pattern_noise(i, n, p) - power_pattern_noise(i - 1, n, p)

#atexit.register(model.save, "/content/drive/My Drive/GG_train_data/saved_model")

selected_train_dataset = train_dataset
noisy = selected_train_dataset
one = noisy

examples_per_jump = selected_train_dataset.shape[0]

flat_shape = (examples_per_jump, PIXELS_PER_IMG)

images = []
noises = []
times = []

noise = np.random.random(size = selected_train_dataset.shape)
noise_diff = noise - one

for jump in range(N_STEPS):
  add_noise = noise_diff * noise_add_frac(jump + 1, N_STEPS)
  
  two = one + add_noise

  images.append(two)
  noises.append(add_noise)
  for i in range(examples_per_jump):
    times.append(np.array([gen_time_matrix(jump, N_STEPS)]))

  one = two

images = np.concatenate(tuple(images), axis = 0)
noises = np.concatenate(tuple(noises), axis = 0)
times = np.concatenate(tuple(times), axis = 0)

#shuffle these up so training and validation splits
#don't have bias
inds = np.random.permutation(images.shape[0])

print("Working with total %d train + valid" % (images.shape[0]))

print("Data shapes", images.shape, noises.shape, times.shape)

images = images[inds]
noises = noises[inds]
times = times[inds]

print(images.shape, noises.shape)

n_total = images.shape[0]

training_frac = 0.8

splitpoint = int(n_total * training_frac)

print("Data shapes", images.shape, noises.shape, times.shape)

#strategy = tf.distribute.experimental.CentralStorageStrategy()

class CustomSaver(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if epoch % 100 == 0:
      self.model.save("model_{}.hd5".format(epoch))

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  if lr < 1e-5:
    return lr
  else:
    return lr * 0.97
  
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

with tf.device('/gpu:0'):
  model = mod_u_net.network_model(TARGET_IMAGE_SIDELENGTH, N_STEPS)

  train = tf.data.Dataset.from_tensor_slices(({'input_1' : images[:splitpoint], 'input_2' : times[:splitpoint]}, noises[:splitpoint])).shuffle(1000)
  valid = tf.data.Dataset.from_tensor_slices(({'input_1' : images[splitpoint:], 'input_2' : times[splitpoint:]}, noises[splitpoint:])).shuffle(1000)

  n_rep = 5
  n_e_per_rep = 40
  offset = 4
  for i in range(n_rep):
    model.fit(train.batch(2 ** (i + offset)), validation_data = valid.batch(2 ** (n_rep + offset - 1)), epochs = n_e_per_rep * (i + 1), initial_epoch = n_e_per_rep * i, shuffle = True, callbacks = [callback])

def generate_test_images(reps):
  noise = np.random.random(size = (4, TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3))
  subtracted = noise
  one = random.randint(0, 1000)
  two = random.randint(0, 1000)
  thr = random.randint(0, 1000)
  fou = random.randint(0, 1000)
  for i in range(reps):
    plt.clf()
    plt.suptitle("Step %d of %d" % (i + 1, N_STEPS))
    pred_noise = model.predict([subtracted, np.array([gen_time_matrix(min(i, N_STEPS - 1), N_STEPS) for j in range(4)])])
    subtracted = subtracted - pred_noise 
    plt.subplot(2, 4, 1)
    plt.imshow(subtracted[0])    
    plt.subplot(2, 4, 2)
    plt.imshow(subtracted[1])    
    plt.subplot(2, 4, 3)
    plt.imshow(subtracted[2])   
    plt.subplot(2, 4, 4)
    plt.imshow(subtracted[3])
    plt.subplot(2, 4, 5)
    plt.imshow(train_dataset[one])    
    plt.subplot(2, 4, 6)
    plt.imshow(train_dataset[two])
    plt.subplot(2, 4, 7)
    plt.imshow(train_dataset[thr])   
    plt.subplot(2, 4, 8)
    plt.imshow(train_dataset[fou])
    if (i + 1) % N_STEPS == 0:
      plt.show()
    else:
      plt.pause(0.1)

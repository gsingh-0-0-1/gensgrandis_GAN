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

from IPython import display

INIT_LR = 1e-3
NOISE_DIM = 1

def contrast_filter(img, b, c, offset):
	img = c / (1 + b ** ( -(img + offset) ) )
	return img

def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(int(TARGET_IMAGE_SIDELENGTH / 2)**2 * 32, use_bias=False, input_shape=(NOISE_DIM,), activation='linear'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	#model.add(layers.Reshape((int(TARGET_IMAGE_SIDELENGTH / 4), int(TARGET_IMAGE_SIDELENGTH / 4), 32)))

	#model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
	model.add(layers.Reshape((int(TARGET_IMAGE_SIDELENGTH / 2), int(TARGET_IMAGE_SIDELENGTH / 2), 32)))
	assert model.output_shape == (None, int(TARGET_IMAGE_SIDELENGTH / 2), int(TARGET_IMAGE_SIDELENGTH / 2), 32)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='linear'))

	#model.add(layers.Flatten())

	#model.add(layers.Dense(TARGET_IMAGE_SIDELENGTH * TARGET_IMAGE_SIDELENGTH * 3, activation='linear'))
	#model.add(layers.Reshape((TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)))
	assert model.output_shape == (None, TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)
	#model.add(layers.BatchNormalization())
	#model.add(layers.LeakyReLU())

	'''model = tf.keras.Sequential([
		layers.Input(shape=(NOISE_DIM,)),
		layers.Dense(int((TARGET_IMAGE_SIDELENGTH / 4) ** 2) * 3, activation='linear'),
		layers.Dense((TARGET_IMAGE_SIDELENGTH ** 2) * 3, activation='linear'),
		layers.Reshape((TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)),
	])'''

	return model


def get_generator_output(inp):
	out = generator(inp, training=False)
	#out = contrast_filter(out, 200, 1, 0)
	return out


def make_discriminator_model():
	model = tf.keras.Sequential([
		layers.Input(shape=(TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)),
		layers.Flatten(),
		#layers.Dense(int((TARGET_IMAGE_SIDELENGTH / 4) ** 2) * 3, activation='linear'),
		#layers.Dense(int((TARGET_IMAGE_SIDELENGTH / 8) ** 2) * 3, activation='linear'),
		#layers.Dense(5, activation='linear'),
		layers.Dense(1, activation='linear'),
	])
	'''model = tf.keras.Sequential()
	model.add(layers.Input(shape=(TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)))

	model.add(layers.Conv2D(64, (10, 10), strides=(2, 2), padding='same'))
	assert model.output_shape == (None, int(TARGET_IMAGE_SIDELENGTH / 2), int(TARGET_IMAGE_SIDELENGTH / 2), 64)
	model.add(layers.LeakyReLU())
	#model.add(layers.Dropout(0.2))

	model.add(layers.Conv2D(1, (2, 2), strides=(2, 2)))
	assert model.output_shape == (None, int(TARGET_IMAGE_SIDELENGTH / 4), int(TARGET_IMAGE_SIDELENGTH / 4), 1)
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2D(32, (2, 2), strides=(2, 2)))
	assert model.output_shape == (None, int(TARGET_IMAGE_SIDELENGTH / 8), int(TARGET_IMAGE_SIDELENGTH / 8), 32)
	model.add(layers.LeakyReLU())

	model.add(layers.Flatten())
	model.add(layers.Dense(100, activation='sigmoid'))
	model.add(layers.Dense(1, activation='linear'))'''
	return model

TARGET_IMAGE_SIDELENGTH = 40

assert TARGET_IMAGE_SIDELENGTH % 8 == 0

train_images = [cv2.cvtColor(cv2.imread("./train_data/" + fname), cv2.COLOR_BGR2RGB)[:TARGET_IMAGE_SIDELENGTH, :TARGET_IMAGE_SIDELENGTH] for fname in os.listdir("./train_data/") if str(fname).startswith('map')]

train_images = np.array(train_images)

train_images = train_images.reshape(train_images.shape[0], TARGET_IMAGE_SIDELENGTH, TARGET_IMAGE_SIDELENGTH, 3)
train_images = (train_images / 255)

train_images = train_images[:5]

print(np.amax(train_images))
BUFFER_SIZE = 60000
BATCH_SIZE = 1

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(INIT_LR)
discriminator_optimizer = tf.keras.optimizers.Adam(INIT_LR)


def generate_seed(shape = (1, NOISE_DIM)):
	return np.random.random(shape) * 10

EPOCHS = int(3e3)
num_examples_to_generate = 4
ROW = 2
COL = 2
seed = generate_seed((num_examples_to_generate, NOISE_DIM))

def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

#@tf.function
def train_step(images):
	noise = generate_seed()

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training=True)

		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)


	g_loss = np.array(gen_loss)
	d_loss = np.array(disc_loss)
	print("\t\tLOSS: ", np.around(g_loss, 5), '\t', np.around(d_loss, 5))

	generator_optimizer.learning_rate = INIT_LR
	discriminator_optimizer.learning_rate = INIT_LR


	#prevent training if one is too far ahead of the other
	calc_g = True
	calc_d = True

	factor = 2

	'''discriminator_optimizer.learning_rate = discriminator_optimizer.learning_rate + (INIT_LR - discriminator_optimizer.learning_rate)/2
	generator_optimizer.learning_rate = generator_optimizer.learning_rate + (INIT_LR - generator_optimizer.learning_rate)/2

	if abs(g_loss - d_loss) > 2:
		if g_loss > d_loss:
			discriminator_optimizer.learning_rate = discriminator_optimizer.learning_rate / factor
			generator_optimizer.learning_rate = generator_optimizer.learning_rate * factor
		if d_loss > g_loss:
			generator_optimizer.learning_rate = generator_optimizer.learning_rate / factor
			discriminator_optimizer.learning_rate = discriminator_optimizer.learning_rate * factor
'''

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

generator = make_generator_model()
discriminator = make_discriminator_model()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_and_save_images(model, epoch, test_input):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = get_generator_output(test_input)

	plt.close()
	plt.figure(figsize=(8, 8))

	for i in range(predictions.shape[0]):
		plt.subplot(ROW, COL, i + 1)
		arr = train_images[epoch % train_images.shape[0]]
		if i != 0:
			arr = np.array(predictions[i])
		#arr = arr - np.amin(arr)
		arr *= 255 / (np.amax(arr))
		arr = np.array(np.around(arr), dtype=int)
		arr[np.where(arr < 0)] = 0
		print(np.amax(arr), np.amin(arr))

		plt.imshow(arr)
		plt.axis('off')

	plt.savefig('gen_imgs/epoch_{:04d}_raw.png'.format(epoch))
	#plt.pause(0.001)
	#plt.show()

def train(dataset, epochs):
	display.clear_output(wait=True)
	generate_and_save_images(generator,
                       0,
                       seed)

	for epoch in range(epochs):
		start = time.time()

		batchnum = 1
		for image_batch in dataset:
			print("\tStarting batch", batchnum, "of epoch", epoch, end=' ')
			train_step(image_batch)
			batchnum += 1

			# Produce images for the GIF as you go
			'''display.clear_output(wait=True)
			generate_and_save_images(generator,
			                         epoch + 1,
			                         seed)'''

		# Save the model every epoch
		#checkpoint.save(file_prefix = checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

		if (epoch + 1) % 1 == 0:
			display.clear_output(wait=True)
			generate_and_save_images(generator,
		                       epoch + 1,
		                       seed)

		if (epoch + 1) % 1 == -1:
			generator.save('gg_map_generator')
			discriminator.save('gg_map_discriminator')

		#generator_optimizer.learning_rate = INIT_LR / (epoch/100 + 1)
		#discriminator_optimizer.learning_rate = INIT_LR / (epoch/100 + 1)

train(train_dataset, EPOCHS)

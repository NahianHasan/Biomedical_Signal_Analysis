import model_compilation as MC
from tensorflow import keras
from functools import partial
import utils as UL
from PIL import Image
import time
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,add,Flatten,Dropout,LeakyReLU,Reshape,Conv2DTranspose,ZeroPadding2D,Add,Conv3DTranspose,Conv3D,ZeroPadding3D,ReLU,UpSampling2D
from tensorflow.keras.layers import MaxPooling2D,MaxPooling3D,UpSampling2D,Cropping2D,concatenate,Input,BatchNormalization,SeparableConv2D,Cropping3D,MaxPooling3D,AveragePooling2D
from tensorflow.keras.initializers import RandomNormal
import config
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import glob
global C,kwargs
C = config.Config()
kwargs = C.kwargs_network

#Input b and g should be 1x1xC
class AdaInstanceNormalization(Layer):
	def __init__(self,
				axis=-1,
				momentum=0.99,
				epsilon=1e-3,
				center=True,
				scale=True,
				**kwargs):
		super(AdaInstanceNormalization, self).__init__(**kwargs)
		self.axis = axis
		self.momentum = momentum
		self.epsilon = epsilon
		self.center = center
		self.scale = scale


	def build(self, input_shape):
		dim = input_shape[0][self.axis]
		if dim is None:
			raise ValueError('Axis ' + str(self.axis) + ' of '
							'input tensor should have a defined dimension '
							'but the layer received an input with shape ' +
							str(input_shape[0]) + '.')

		super(AdaInstanceNormalization, self).build(input_shape)

	def call(self, inputs, training=None):
		input_shape = tf.keras.backend.int_shape(inputs[0])
		reduction_axes = list(range(0, len(input_shape)))

		beta = inputs[1]
		gamma = inputs[2]

		if self.axis is not None:
			del reduction_axes[self.axis]

		del reduction_axes[0]
		mean = tf.keras.backend.mean(inputs[0], reduction_axes, keepdims=True)
		stddev = tf.keras.backend.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
		normed = (inputs[0] - mean) / stddev

		return normed * gamma + beta

	def get_config(self):
		config = {
			'axis': self.axis,
			'momentum': self.momentum,
			'epsilon': self.epsilon,
			'center': self.center,
			'scale': self.scale
		}
		base_config = super(AdaInstanceNormalization, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def compute_output_shape(self, input_shape):
		return input_shape[0]
#Input b and g should be HxWxC
class SPADE(Layer):
	def __init__(self,
				axis=-1,
				momentum=0.99,
				epsilon=1e-3,
				center=True,
				scale=True,
				**kwargs):
			super(SPADE, self).__init__(**kwargs)
			self.axis = axis
			self.momentum = momentum
			self.epsilon = epsilon
			self.center = center
			self.scale = scale


	def build(self, input_shape):
		dim = input_shape[0][self.axis]
		if dim is None:
			raise ValueError('Axis ' + str(self.axis) + ' of '
							'input tensor should have a defined dimension '
							'but the layer received an input with shape ' +
							str(input_shape[0]) + '.')
		super(SPADE, self).build(input_shape)

	def call(self, inputs, training=None):
		input_shape = tf.keras.backend.int_shape(inputs[0])

		beta = inputs[1]
		gamma = inputs[2]

		reduction_axes = [0, 1, 2]
		mean = tf.keras.backend.mean(inputs[0], reduction_axes, keepdims=True)
		stddev = tf.keras.backend.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
		normed = (inputs[0] - mean) / stddev

		return normed * gamma + beta

	def get_config(self):
		config = {
			'axis': self.axis,
			'momentum': self.momentum,
			'epsilon': self.epsilon,
			'center': self.center,
			'scale': self.scale
		}
		base_config = super(SPADE, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def compute_output_shape(self, input_shape):
		return input_shape[0]

#Style Z
def noise(n,latent_dimension=C.latent_dimension):
	return np.random.normal(0.0, 1.0, size = [n, latent_dimension])
#Noise Sample
def noiseImage(n,image_shape=C.image_shape):
	return np.random.uniform(0.0, 1.0, size = [n, image_shape[0], image_shape[1], 1])
#Get random samples from an array
def get_rand(array, amount):
	idx = np.random.randint(0, array.shape[0], amount)
	return array[idx]
#r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight,sample_weight=None):
	gradients = tf.keras.backend.gradients(y_pred, averaged_samples)[0]
	gradients_sqr = tf.keras.backend.square(gradients)
	gradient_penalty = tf.keras.backend.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

	# weight * ||grad||^2
	# Penalize the gradient norm
	return tf.keras.backend.mean(gradient_penalty * weight)
#Upsample, Convolution, AdaIN, Noise, Activation, Convolution, AdaIN, Noise, Activation
def g_block(inp, style, noise, fil, u = True):
	b = Dense(fil)(style)
	b = Reshape([1, 1, fil])(b)
	g = Dense(fil)(style)
	g = Reshape([1, 1, fil])(g)

	n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)

	if u:
		out = UpSampling2D(interpolation = 'bilinear')(inp)
		out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
	else:
		out = Activation('linear')(inp)

	out = AdaInstanceNormalization()([out, b, g])
	out = add([out, n])
	out = LeakyReLU(0.01)(out)

	b = Dense(fil)(style)
	b = Reshape([1, 1, fil])(b)
	g = Dense(fil)(style)
	g = Reshape([1, 1, fil])(g)

	n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)

	out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
	out = AdaInstanceNormalization()([out, b, g])
	out = add([out, n])
	out = LeakyReLU(0.01)(out)

	return out
#Convolution, Activation, Pooling, Convolution, Activation
def d_block(inp, fil, p = True):
	route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
	route2 = LeakyReLU(0.01)(route2)
	if p:
		route2 = AveragePooling2D()(route2)
	route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(route2)
	out = LeakyReLU(0.01)(route2)

	return out

#This object holds the models
class GAN(object):
	def __init__(self, lr = C.initial_lrate):
		#Models
		self.D = None
		self.G = None

		self.DM = None
		self.AM = None

		#Config
		self.LR = lr
		self.steps = 1

		#Init Models
		self.discriminator()
		self.generator()
	def discriminator(self):
		if self.D:
			return self.D

		inp = Input(shape = [C.image_shape[0], C.image_shape[1], C.image_channels])

		# Size
		x = d_block(inp, 16) #Size / 2
		x = d_block(x, 32) #Size / 4
		x = d_block(x, 64) #Size / 8

		if (C.image_shape[0] > 32):
			x = d_block(x, 128) #Size / 16

		if (C.image_shape[0] > 64):
			x = d_block(x, 192) #Size / 32

		if (C.image_shape[0] > 128):
			x = d_block(x, 256) #Size / 64

		if (C.image_shape[0] > 256):
			x = d_block(x, 384) #Size / 128

		if (C.image_shape[0] > 512):
			x = d_block(x, 512) #Size / 256

		x = Flatten()(x)

		x = Dense(128)(x)
		x = Activation('relu')(x)

		x = Dropout(0.1)(x)
		x = Dense(1)(x)

		self.D = Model(inputs = inp, outputs = x)

		return self.D
	def generator(self):
		if self.G:
			return self.G

		#Style FC, I only used 2 fully connected layers instead of 8 for faster training
		inp_s = Input(shape = [C.latent_dimension])
		sty = Dense(512, kernel_initializer = 'he_normal')(inp_s)
		sty = LeakyReLU(0.1)(sty)
		sty = Dense(512, kernel_initializer = 'he_normal')(sty)
		sty = LeakyReLU(0.1)(sty)

		#Get the noise image and crop for each size
		inp_n = Input(shape = [C.image_shape[0], C.image_shape[1], 1])
		noi = [Activation('linear')(inp_n)]
		curr_size = C.image_shape[0]
		while curr_size > 4:
			curr_size = int(curr_size / 2)
			noi.append(Cropping2D(int(curr_size/2))(noi[-1]))

		#Here do the actual generation stuff
		inp = Input(shape = [1])
		x = Dense(4 * 4 * 512, kernel_initializer = 'he_normal')(inp)
		x = Reshape([4, 4, 512])(x)
		x = g_block(x, sty, noi[-1], 512, u=False)

		if(C.image_shape[0] >= 1024):
			x = g_block(x, sty, noi[7], 512) # Size / 64
		if(C.image_shape[0] >= 512):
			x = g_block(x, sty, noi[6], 384) # Size / 64
		if(C.image_shape[0] >= 256):
			x = g_block(x, sty, noi[5], 256) # Size / 32
		if(C.image_shape[0] >= 128):
			x = g_block(x, sty, noi[4], 192) # Size / 16
		if(C.image_shape[0] >= 64):
			x = g_block(x, sty, noi[3], 128) # Size / 8

		x = g_block(x, sty, noi[2], 64) # Size / 4
		x = g_block(x, sty, noi[1], 32) # Size / 2
		x = g_block(x, sty, noi[0], 16) # Size

		x = Conv2D(filters = C.image_channels, kernel_size = 1, padding = 'same', activation = C.generator_activation_function)(x)

		self.G = Model(inputs = [inp_s, inp_n, inp], outputs = x)

		return self.G
	def AdModel(self):
		#D does not update
		self.D.trainable = False
		for layer in self.D.layers:
			layer.trainable = False

		#G does update
		self.G.trainable = True
		for layer in self.G.layers:
			layer.trainable = True

		#This model is simple sequential one with inputs and outputs
		gi = Input(shape = [C.latent_dimension])
		gi2 = Input(shape = [C.image_shape[0], C.image_shape[1], 1])
		gi3 = Input(shape = [1])

		gf = self.G([gi, gi2, gi3])
		df = self.D(gf)

		self.AM = Model(inputs = [gi, gi2, gi3], outputs = df)
		self.AM = MC.Model_Compile(self.AM,C.gan_optimizer,'mse')
		return self.AM
	def DisModel(self):
		#D does update
		self.D.trainable = True
		for layer in self.D.layers:
			layer.trainable = True

		#G does not update
		self.G.trainable = False
		for layer in self.G.layers:
			layer.trainable = False

		# Real Pipeline
		ri = Input(shape = [C.image_shape[0], C.image_shape[1], C.image_channels])
		dr = self.D(ri)

		# Fake Pipeline
		gi = Input(shape = [C.latent_dimension])
		gi2 = Input(shape = [C.image_shape[0], C.image_shape[1], 1])
		gi3 = Input(shape = [1])
		gf = self.G([gi, gi2, gi3])
		df = self.D(gf)

		# Samples for gradient penalty
		# For r1 use real samples (ri)
		# For r2 use fake samples (gf)
		da = self.D(ri)

		# Model With Inputs and Outputs
		self.DM = Model(inputs=[ri, gi, gi2, gi3], outputs=[dr, df, da])

		# Create partial of gradient penalty loss
		# For r1, averaged_samples = ri
		# For r2, averaged_samples = gf
		# Weight of 10 typically works
		partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 5)

		#Compile With Corresponding Loss Functions
		self.DM = MC.Model_Compile(self.DM,C.discriminator_optimizer,['mse', 'mse', partial_gp_loss])
		return self.DM

class WGAN(object):
	def __init__(self, steps = -1, lr = C.initial_lrate, silent = True):
		self.GAN = GAN(lr = lr)
		self.DisModel = self.GAN.DisModel()
		self.AdModel = self.GAN.AdModel()
		self.generator = self.GAN.generator()

		if steps >= 0:
			self.GAN.steps = steps

		self.lastblip = time.process_time()

		self.noise_level = 0
		self.silent = silent

		#Train Generator to be in the middle, not all the way at real. Apparently works better??
		self.nones = -np.ones((C.nb_batches, 1), dtype=np.float32)

		self.enoise = noise(8)
		self.enoiseImage = noiseImage(8)
		UL.prepare_subject_list(C.data_directory)#Prepare subject List
		self.dataset=UL.load_records(C.data_list)#Load the subject List

	def train_dis(self):
		#Get Data
		X,Y = UL.generate_real_samples(self.dataset, C.nb_batches)
		#print(X.shape)
		#print(Y.shape)
		#print(noise(C.nb_batches).shape)
		#print(noiseImage(C.nb_batches).shape)
		train_data = [X, noise(C.nb_batches), noiseImage(C.nb_batches), Y]
		#Train
		d_loss = self.DisModel.train_on_batch(train_data, [Y, self.nones, Y])
		return d_loss
	def train_gen(self):
		#Train
		ones = np.ones((C.nb_batches, 1), dtype=np.float32)
		zeros = np.zeros((C.nb_batches, 1), dtype=np.float32)
		g_loss = self.AdModel.train_on_batch([noise(C.nb_batches), noiseImage(C.nb_batches), ones], zeros)
		return g_loss
	def evaluate(self, num = 0, trunc = 2.0): #8x4 images, bottom row is constant

		n = noise(1)
		n2 = noiseImage(1)

		im2 = self.generator.predict([n, n2, np.ones([1, 1])])
		im3 = self.generator.predict([self.enoise, self.enoiseImage, np.ones([8, 1])])
		if C.generator_activation_function == 'tanh':
			im2 = (im2 + 1)/2

		#r12 = np.concatenate(im2[:8], axis = 1)
		#r22 = np.concatenate(im2[8:16], axis = 1)
		#r32 = np.concatenate(im2[16:24], axis = 1)
		#r43 = np.concatenate(im3[:8], axis = 1)

		#c1 = np.concatenate([r12, r22, r32, r43], axis = 0)
		#c1 = np.reshape(c1,(c1.shape[0], c1.shape[1]))
		#x = Image.fromarray(np.uint8(im2))

		im2 = np.reshape(im2,[im2.shape[1],im2.shape[2],im2.shape[3]])
		im2 = im2 * 255
		np.savez("Results/i"+str(num)+".npz",im2)
		#plt.imshow(im2)
		#plt.axis('off')
		#plt.title(C.tissue_map[str(max(C.nb_segments))])
		#plt.show()
		#plt.savefig("Results/i"+str(num)+".jpg")
	def evaluate2(self, s1, s2, n1, n2, num = 0, weight = 0.5):

		s = normalize((s2 * weight) + (s1 * (1 - weight)))
		n = (n2 * weight) + (n1 * (1 - weight))

		im2 = self.generator.predict([s, n, np.ones([32, 1])])

		r12 = np.concatenate(im2[:8], axis = 1)
		r22 = np.concatenate(im2[8:16], axis = 1)
		r32 = np.concatenate(im2[16:24], axis = 1)
		r43 = np.concatenate(im2[24:], axis = 1)

		c1 = np.concatenate([r12, r22, r32, r43], axis = 0)
		c1 = np.reshape(c1,(c1.shape[0], c1.shape[1]))
		x = Image.fromarray(np.uint8(c1))

		x.save("Results/i"+str(num)+".jpg")
	def evalTrunc(self, num = 0, trunc = 1.8):

		n = np.clip(noise(16), -trunc, trunc)
		n2 = noiseImage(16)

		im2 = self.generator.predict([n, n2, np.ones([16, 1])])

		r12 = np.concatenate(im2[:4], axis = 1)
		r22 = np.concatenate(im2[4:8], axis = 1)
		r32 = np.concatenate(im2[8:12], axis = 1)
		r43 = np.concatenate(im2[12:], axis = 1)

		c1 = np.concatenate([r12, r22, r32, r43], axis = 0)
		c1 = np.reshape(c1,(c1.shape[0], c1.shape[1]))
		x = Image.fromarray(np.uint8(c1))

		x.save("Results/t"+str(num)+".jpg")
	def saveModel(self, model, name, num): #Save a Model
		json = model.to_json()
		with open("Models/"+name+".json", "w") as json_file:
			json_file.write(json)

		model.save_weights("Models/"+name+"_"+str(num)+".hdf5")
	def loadModel(self, name, num): #Load a Model
		#load_architecture
		json_file = open("Models/"+name+".json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		#model = model_from_json(loaded_model_json)

		model = model_from_json(loaded_model_json, custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
		model.load_weights("Models/"+name+"_model_"+str(num)+".hdf5")

		return model
	def save(self, num): #Save JSON and Weights into /Models/
		self.saveModel(self.GAN.G, "gen", num)
		self.saveModel(self.GAN.D, "dis", num)
	def load(self, num): #Load JSON and Weights from /Models/
		steps1 = self.GAN.steps

		self.GAN = None
		self.GAN = GAN()

		#Load Models
		self.GAN.G = self.loadModel("generator", num)
		self.GAN.D = self.loadModel("discriminator", num)

		self.GAN.steps = steps1

		self.generator = self.GAN.generator()
		self.DisModel = self.GAN.DisModel()
		self.AdModel = self.GAN.AdModel()
	def train(self):
		# train the generator and discriminator
		bat_per_epo = int(len(self.dataset) / C.nb_batches)
		half_batch = int(C.nb_batches / 2)
		# calculate the total iterations based on batch and epoch
		if C.resume_status:
			initial_step = int(C.initial_iter)
		else:
			initial_step = 0
		n_steps = bat_per_epo * C.nb_epochs

		# manually enumerate epochs
		for i in range(initial_step,n_steps):
			#Train Alternating
			a = self.train_dis()
			b = self.train_gen()
			print('step = ',i,' Discriminator : [', end=' ')
			for p in range(0,len(a)):
				print(self.DisModel.metrics_names[p],':',a[p], end=' ')
			print('] Generator : [', end=' ')
			for q in range(0,len(b)):
				print(self.AdModel.metrics_names[q],':',b[q], end=' ')
			print('')
			if (i+1) % C.nb_evals == 0:
				#Save Model
				UL.save_model(self.GAN.G,'./Models/generator.json','./Models/generator_model_%03d.hdf5'% (i + 1))
				UL.save_model(self.GAN.D,'./Models/discriminator.json','./Models/discriminator_model_%03d.hdf5'% (i + 1))
				print('Step = ',i, ' Model Saved')




config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
'''
model = WGAN()
print(model.DisModel.summary())
print(model.AdModel.summary())
print(model.generator.summary())
if C.resume_status:
	model.load(C.initial_iter)
model.train()
'''


#Test
for i in range(11500,251751,50):
	model = WGAN()
	print('Loaded Model = ',i)
	model.load(i)
	model.evaluate(i)
	tf.keras.backend.clear_session()

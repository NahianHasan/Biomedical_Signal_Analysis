import model_compilation as MC
import GAN_evaluation as GE
from tensorflow import keras
from functools import partial
import utils as UL
from PIL import Image
import folder_creation as FC
import time,os,glob,argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,Conv3D,add,Flatten,Dropout,LeakyReLU,Reshape,Conv2DTranspose,ZeroPadding2D,Add,Conv3DTranspose,Conv3D,ZeroPadding3D,ReLU,Embedding,AveragePooling3D
from tensorflow.keras.layers import MaxPooling2D,MaxPooling3D,UpSampling2D,UpSampling3D,Cropping2D,Cropping3D,Concatenate,Input,BatchNormalization,SeparableConv2D,Cropping3D,MaxPooling3D,AveragePooling2D
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K
import config
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import glob
global C,kwargs
C = config.Config()


def activation_Quantizatioin(x):
	sigmoid = K.math.sigmoid(x)
	input_data = K.where(K.math.less(input_data,0.1),[0],sigmoid)
	input_data = K.where(K.math.logical_and(K.math.greater_equal(input_data,0.1),K.math.less(input_data,0.3)),[0.2],input_data)
	input_data = K.where(K.math.logical_and(K.math.greater_equal(input_data,0.3),K.math.less(input_data,0.5)),[0.4],input_data)
	input_data = K.where(K.math.logical_and(K.math.greater_equal(input_data,0.5),K.math.less(input_data,0.7)),[0.6],input_data)
	input_data = K.where(K.math.logical_and(K.math.greater_equal(input_data,0.7),K.math.less(input_data,0.9)),[0.8],input_data)
	input_data = K.where(K.math.greater_equal(input_data,0.9),[1.0],input_data)
	return input_data
class layer_Quantization(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(Quantization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.kernel = self.add_weight(name = 'kernel',shape = (input_shape[1], self.output_dim),initializer = 'normal', trainable = True)
		super(Quantization, self).build(input_shape) ##Be sure to call this at the end
	def call(self, input_data):
		input_data = K.where(K.math.less(input_data,0.1),[0],input_data)
		input_data = K.where(K.math.logical_and(K.math.greater_equal(input_data,0.1),K.math.less(input_data,0.3)),[0.2],input_data)
		input_data = K.where(K.math.logical_and(K.math.greater_equal(input_data,0.3),K.math.less(input_data,0.5)),[0.4],input_data)
		input_data = K.where(K.math.logical_and(K.math.greater_equal(input_data,0.5),K.math.less(input_data,0.7)),[0.6],input_data)
		input_data = K.where(K.math.logical_and(K.math.greater_equal(input_data,0.7),K.math.less(input_data,0.9)),[0.8],input_data)
		input_data = K.where(K.math.greater_equal(input_data,0.9),[1.0],input_data)
		return input_data
	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)
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
		input_shape = K.int_shape(inputs[0])
		reduction_axes = list(range(0, len(input_shape)))

		beta = inputs[1]
		gamma = inputs[2]

		if self.axis is not None:
			del reduction_axes[self.axis]

		del reduction_axes[0]
		mean = K.mean(inputs[0], reduction_axes, keepdims=True)
		stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
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
		input_shape = K.int_shape(inputs[0])

		beta = inputs[1]
		gamma = inputs[2]

		reduction_axes = [0, 1, 2]
		mean = K.mean(inputs[0], reduction_axes, keepdims=True)
		stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
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
	return np.random.uniform(0.0, 1.0, size = [n]+image_shape+[1])
#Get random samples from an array
def get_rand(array, amount):
	idx = np.random.randint(0, array.shape[0], amount)
	return array[idx]
#r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight,sample_weight=None):
	gradients = K.gradients(y_pred, averaged_samples)[0]
	gradients_sqr = K.square(gradients)
	gradient_penalty = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

	# weight * ||grad||^2
	# Penalize the gradient norm
	return K.mean(gradient_penalty * weight)
#Upsample, Convolution, AdaIN, Noise, Activation, Convolution, AdaIN, Noise, Activation
def g_block_3D(dict):
	inp, style, noise, fil, u, real_im = [dict[str(i)] for i in range(1,7)]
	fil = int(fil*C.filter_scale/100)
	b = Dense(fil)(style)
	b = Reshape([1, 1, 1, fil])(b)
	g = Dense(fil)(style)
	g = Reshape([1, 1, 1, fil])(g)

	if real_im:
		n = Conv3D(filters = int(np.ceil(fil/2)), kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
		n1 = Conv3D(filters = int(np.ceil(fil/2)), kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(real_im)
	else:
		n = Conv3D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)

	if u:
		out = UpSampling3D(interpolation = 'bilinear')(inp)
		out = Conv3D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
	else:
		out = Activation('linear')(inp)

	n = Concatenate(axis=C.channel_pos)([n, n1]) if real_im is not None else n
	out = add([out, n])
	out = AdaInstanceNormalization()([out, b, g])
	out = LeakyReLU(0.01)(out)

	out = Conv3D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
	out = add([out, n])
	out = AdaInstanceNormalization()([out, b, g])
	out = LeakyReLU(0.01)(out)

	return out
def g_block(dict):
	inp, style, noise, fil, u, real_im = [dict[str(i)] for i in range(1,7)]
	fil = int(fil*C.filter_scale/100)
	b = Dense(fil)(style)
	b = Reshape([1, 1, fil])(b)
	g = Dense(fil)(style)
	g = Reshape([1, 1, fil])(g)

	if real_im:
		n = Conv2D(filters = int(np.ceil(fil/2)), kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
		n1 = Conv2D(filters = int(np.ceil(fil/2)), kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(real_im)
	else:
		n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)

	if u:
		out = UpSampling2D(interpolation = 'bilinear')(inp)
		out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
	else:
		out = Activation('linear')(inp)

	n = Concatenate(axis=C.channel_pos)([n, n1]) if real_im is not None else n
	out = add([out, n])
	out = AdaInstanceNormalization()([out, b, g])
	out = LeakyReLU(0.01)(out)

	out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
	out = add([out, n])
	out = AdaInstanceNormalization()([out, b, g])
	out = LeakyReLU(0.01)(out)

	return out
#Convolution, Activation, Pooling, Convolution, Activation
def d_block_3D(inp, fil, p = True):
	fil = int(fil*C.filter_scale/100)
	route2 = Conv3D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
	route2 = LeakyReLU(0.01)(route2)
	if p:
		route2 = AveragePooling3D()(route2)
	route2 = Conv3D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(route2)
	out = LeakyReLU(0.01)(route2)

	return out
def d_block(inp, fil, p = True):
	fil = int(fil*C.filter_scale/100)
	route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
	route2 = LeakyReLU(0.01)(route2)
	if p:
		route2 = AveragePooling2D()(route2)
	route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(route2)
	out = LeakyReLU(0.01)(route2)

	return out

#This object holds the models
class GAN(object):
	def __init__(self, folder, lr = C.initial_lrate):
		#Models
		self.D = None
		self.G = None

		self.DM = None
		self.AM = None

		#Config
		self.LR = lr
		self.steps = 1
		self.folder=folder

		#Init Models
		self.discriminator()
		self.generator()
	def discriminator(self):
		if self.D:
			return self.D

		inp = Input(shape = C.image_shape + C.image_channels)
		if C.conditional_GAN:
			# label input
			in_label = Input(shape=(1,))
			# embedding for categorical input
			li = Embedding(C.conditional_classes, 50)(in_label)
			# scale up to image dimensions with linear activation
			n_nodes = np.prod(C.image_shape)
			li = Dense(n_nodes)(li)
			# reshape to additional channel
			li = Reshape(tuple(C.image_shape + [1]))(li)
			# concat label as a channel
			x = Concatenate()([inp, li])

		# Size
		x = d_block(inp, 16) if not C.train_3D else d_block_3D(inp, 16)#Size / 2
		x = d_block(x, 32) if not C.train_3D else d_block_3D(inp, 32)#Size / 4
		x = d_block(x, 64) if not C.train_3D else d_block_3D(inp, 64)#Size / 8

		if (C.image_shape[0] > 32):
			x = d_block(x, 128) if not C.train_3D else d_block_3D(x, 128)#Size / 16

		if (C.image_shape[0] > 64):
			x = d_block(x, 192) if not C.train_3D else d_block_3D(x, 192)#Size / 32

		if (C.image_shape[0] > 128):
			x = d_block(x, 256) if not C.train_3D else d_block_3D(x, 256)#Size / 64

		if (C.image_shape[0] > 256):
			x = d_block(x, 384) if not C.train_3D else d_block_3D(x, 384)#Size / 128

		if (C.image_shape[0] > 512):
			x = d_block(x, 512) if not C.train_3D else d_block_3D(x, 512)#Size / 256

		x = Flatten()(x)

		x = Dense(128)(x)
		x = Activation('relu')(x)

		x = Dropout(0.1)(x)
		x = Dense(1)(x)

		if C.conditional_GAN:
			self.D = Model(inputs = [inp,in_label], outputs = x)
		else:
			self.D = Model(inputs = inp, outputs = x)
		plot_model(self.D, to_file=self.folder+'/Model_Figures/discriminator.png', show_shapes=True, show_layer_names=True)
		return self.D
	def generator(self):
		if self.G:
			return self.G

		inp_s = Input(shape = [C.latent_dimension])
		sty = Dense(512, kernel_initializer = 'he_normal')(inp_s)
		sty = LeakyReLU(0.1)(sty)
		for i in range(1,9):
			sty = Dense(512, kernel_initializer = 'he_normal')(sty)
			sty = LeakyReLU(0.1)(sty)

		#Get the noise image and crop for each size
		inp_n = Input(shape = C.image_shape+[1])
		noi = [Activation('linear')(inp_n)]
		curr_size = C.image_shape[0]
		while curr_size > 4:
			curr_size = int(curr_size / 2)
			if C.train_3D:
				noi.append(Cropping3D(int(curr_size/2))(noi[-1]))
			else:
				noi.append(Cropping2D(int(curr_size/2))(noi[-1]))

		if C.Train_as_SeedGAN:
			#Get the real image for SeedGAN training
			inp_re = Input(shape = C.image_shape+C.image_channels)
			real_im = [Activation('linear')(inp_re)]
			curr_size = C.image_shape[0]
			while curr_size > 4:
				curr_size = int(curr_size / 2)#rescale to this size
				dim = tf.cast(tf.math.scalar_mul(0.5,tf.cast(tf.shape(real_im[-1],out_type=tf.int32),dtype=tf.float32)),dtype=tf.int32)[1:-1]
				real_im.append(tf.image.resize(real_im[-1], dim, method='bilinear'))

		#Here do the actual generation stuff
		inp = Input(shape = [1])
		if C.train_3D:
			x = Dense(4 * 4 * 4 * int(512*C.filter_scale/100), kernel_initializer = 'he_normal')(inp)
			x = Reshape([4, 4, 4, int(512*C.filter_scale/100)])(x)
		else:
			x = Dense(4 * 4 * int(512*C.filter_scale/100), kernel_initializer = 'he_normal')(inp)
			x = Reshape([4, 4, int(512*C.filter_scale/100)])(x)

		#Add embedding layer for conditional GAN
		if C.conditional_GAN:
			# label input
			in_label = Input(shape=(1,))
			# embedding for categorical input
			li = Embedding(C.conditional_classes, 50)(in_label)
			# scale up to image dimensions with linear activation
			n_nodes = 4 * 4 * 4 if C.train_3D else 4 * 4
			li = Dense(n_nodes)(li)
			# reshape to additional channel
			li = Reshape((4,4,1))(li) if C.train_3D else Reshape((4,4,4,1))(li)
			x = Concatenate()([x, li])

		data_block = {'1':x, '2':sty, '3':noi[-1], '4':512, '5':True, '6':real_im[-1]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[-1], '4':512, '5':False, '6':None}
		x = g_block(data_block) if not C.train_3D else g_block_3D(data_block)#First block, there is no upsampling

		if(C.image_shape[0] >= 1024):
			data_block = {'1':x, '2':sty, '3':noi[7], '4':512, '5':False, '6':real_im[7]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[7], '4':512, '5':True, '6':None}
			x = g_block(data_block) if not C.train_3D else g_block_3D(data_block)# Size / 64
		if(C.image_shape[0] >= 512):
			data_block = {'1':x, '2':sty, '3':noi[6], '4':384, '5':False, '6':real_im[6]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[6], '4':384, '5':True, '6':None}
			x = g_block(data_block) if not C.train_3D else g_block_3D(data_block)# Size / 64
		if(C.image_shape[0] >= 256):
			data_block = {'1':x, '2':sty, '3':noi[5], '4':256, '5':False, '6':real_im[5]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[5], '4':256, '5':True, '6':None}
			x = g_block(data_block) if not C.train_3D else g_block_3D(data_block)# Size / 32
		if(C.image_shape[0] >= 128):
			data_block = {'1':x, '2':sty, '3':noi[4], '4':192, '5':False, '6':real_im[4]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[4], '4':192, '5':True, '6':None}
			x = g_block(data_block) if not C.train_3D else g_block_3D(data_block)# Size / 16
		if(C.image_shape[0] >= 64):
			data_block = {'1':x, '2':sty, '3':noi[3], '4':128, '5':False, '6':real_im[3]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[3], '4':128, '5':True, '6':None}
			x = g_block(data_block) if not C.train_3D else g_block_3D(data_block)# Size / 8

		data_block = {'1':x, '2':sty, '3':noi[2], '4':64, '5':False, '6':real_im[2]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[2], '4':64, '5':True, '6':None}
		x = g_block(data_block) if not C.train_3D else g_block_3D(data_block)# Size / 4
		data_block = {'1':x, '2':sty, '3':noi[1], '4':32, '5':False, '6':real_im[1]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[1], '4':32, '5':True, '6':None}
		x = g_block(data_block) if not C.train_3D else g_block_3D(data_block) # Size / 2
		data_block = {'1':x, '2':sty, '3':noi[0], '4':16, '5':False, '6':real_im[0]} if C.Train_as_SeedGAN else {'1':x, '2':sty, '3':noi[0], '4':16, '5':True, '6':None}
		x = g_block(data_block) if not C.train_3D else g_block_3D(data_block) # Size

		if C.train_3D:
			x = Conv3D(filters = C.image_channels[0], kernel_size = 1, padding = 'same', activation = C.generator_activation_function)(x)
		else:
			x = Conv2D(filters = C.image_channels[0], kernel_size = 1, padding = 'same', activation = C.generator_activation_function)(x)

		inputs=[inp_s, inp_n, inp, inp_re] if C.Train_as_SeedGAN else [inp_s, inp_n, inp]
		if C.conditional_GAN:
			inputs.append(in_label)

		self.G = Model(inputs = inputs, outputs = x)
		plot_model(self.G, to_file=self.folder+'/Model_Figures/generator.png', show_shapes=True, show_layer_names=True)
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
		gi2 = Input(shape = C.image_shape + [1])
		gi3 = Input(shape = [1])
		if C.Train_as_SeedGAN:
			gi5 = Input(shape = C.image_shape + C.image_channels)
			inputs_g = [gi,gi2,gi3,gi5]
		else:
			inputs_g = [gi,gi2,gi3]

		if C.conditional_GAN:
			gi4 = Input(shape = [1])
			inputs_g.append(gi4)

		gf = self.G(inputs_g)
		if C.conditional_GAN:
			inputs_d = [gf,gi4]
		else:
			inputs_d = gf
		df = self.D(inputs_d)
		self.AM = Model(inputs = inputs_g, outputs = df)
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

		# Fake Pipeline
		gi = Input(shape = [C.latent_dimension])
		gi2 = Input(shape = C.image_shape + [1])
		gi3 = Input(shape = [1])
		if C.Train_as_SeedGAN:
			gi5 = Input(shape = C.image_shape + C.image_channels)
			inputs_g = [gi,gi2,gi3,gi5]
		else:
			inputs_g = [gi,gi2,gi3]

		if C.conditional_GAN:
			gi4 = Input(shape = [1])
			inputs_g.append(gi4)

		gf = self.G(inputs_g)
		if C.conditional_GAN:
			inputs_d = [gf,gi4]
		else:
			inputs_d = gf
		df = self.D(inputs_d)

		# Real Pipeline
		ri = Input(shape = C.image_shape + C.image_channels)
		if C.conditional_GAN:
			dr = self.D([ri,gi4])
			# Samples for gradient penalty
			# For r1 use real samples (ri)
			# For r2 use fake samples (gf)
			da = self.D([ri,gi4])
		else:
			dr = self.D(ri)
			da = self.D(ri)

		# Model With Inputs and Outputs
		inputs = [ri] + inputs_g

		self.DM = Model(inputs=inputs, outputs=[dr, df, da])

		# Create partial of gradient penalty loss
		# For r1, averaged_samples = ri
		# For r2, averaged_samples = gf
		# Weight of 10 typically works
		partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 5)

		#Compile With Corresponding Loss Functions
		self.DM = MC.Model_Compile(self.DM,C.discriminator_optimizer,['mse', 'mse', partial_gp_loss])
		return self.DM
class WGAN(object):
	def __init__(self, folder, steps = -1, lr = C.initial_lrate, silent = True):
		self.folder = folder

		self.GAN = GAN(self.folder,lr = lr)
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
		if C.Train_as_SeedGAN:
			train_data = [X, noise(C.nb_batches), noiseImage(C.nb_batches), Y, X]
		else:
			train_data = [X, noise(C.nb_batches), noiseImage(C.nb_batches), Y]

		if C.conditional_GAN:
			train_data.append(Y)

		#Train
		d_loss = self.DisModel.train_on_batch(train_data, [Y, self.nones, Y])
		return d_loss
	def train_gen(self):
		#Train
		ones = np.ones((C.nb_batches, 1), dtype=np.float32)
		zeros = np.zeros((C.nb_batches, 1), dtype=np.float32)
		if C.Train_as_SeedGAN:
			X,_ = UL.generate_real_samples(self.dataset, C.nb_batches)
			train_data = [noise(C.nb_batches), noiseImage(C.nb_batches), ones, X]
		else:
			train_data = [noise(C.nb_batches), noiseImage(C.nb_batches), ones]
		if C.conditional_GAN:
			labels=randint(0, C.conditional_classes, C.nb_batches)
			train_data.append(labels)

		g_loss = self.AdModel.train_on_batch(train_data, ones)

		return g_loss
	def evaluate(self, ep, save_folder, num = 1, trunc = 2.0, verbose=0, format='jpg'): #8x4 images, bottom row is constant

		for i in range(0,num):
			n = noise(1)
			n2 = noiseImage(1)
			if C.Train_as_SeedGAN:
				X,_ = UL.generate_real_samples(self.dataset, 1)
				data = [n, n2, np.ones([1, 1]),X]
			else:
				data = [n, n2, np.ones([1, 1])]

			if C.conditional_GAN:
				label = randint(0, C.conditional_classes, 1)
				data.append(label)

			im2 = self.generator.predict(data)
			if C.generator_activation_function == 'tanh':
				im2 = (im2 + 1)/2

			#print(im2.shape)
			im2 = np.reshape(im2,list(im2.shape))
			im2 = im2 * 255



			#Start Quantizing
			if verbose:
				print('saved image id = ',i,end = "\r",flush=True)
			#Quantize the images
			img=im2
			hist=np.histogram(img,bins=6)
			pairs=sorted(zip(hist[0],hist[1]))
			tuples=zip(*pairs)
			list1,list2=[list(tuple) for tuple in tuples]
			list2.sort()

			img=np.array(img)
			img[np.where(img < list2[-6]+(list2[-5]-list2[-6])/2)] = 0.0
			img[np.where((img >= list2[-6]+(list2[-5]-list2[-6])/2) & (img < list2[-5]+(list2[-4]-list2[-5])/2))] = list2[-5]
			img[np.where((img >= list2[-5]+(list2[-4]-list2[-5])/2) & (img < list2[-4]+(list2[-3]-list2[-4])/2))] = list2[-4]
			img[np.where((img >= list2[-4]+(list2[-3]-list2[-4])/2) & (img < list2[-3]+(list2[-2]-list2[-3])/2))] = list2[-3]
			img[np.where((img >= list2[-3]+(list2[-2]-list2[-3])/2) & (img < list2[-2]+(list2[-1]-list2[-2])/2))] = list2[-2]
			img[np.where(img >= list2[-2]+(list2[-1]-list2[-2])/2)] = 255.0

			im_ind = np.unique(img)
			im_ind.sort()
			img[np.where(img == im_ind[1])] = 51.123456
			img[np.where(img == im_ind[2])] = 102.123456
			img[np.where(img == im_ind[3])] = 153.123456
			img[np.where(img == im_ind[4])] = 204.123456
			im_ind = np.unique(img)
			im_ind.sort()
			img[np.where(img == im_ind[1])] = 51.0
			img[np.where(img == im_ind[2])] = 102.0
			img[np.where(img == im_ind[3])] = 153.0
			img[np.where(img == im_ind[4])] = 204.0
			img = np.reshape(img,tuple(img.shape[:-1]))
			if format.upper()=='JPG':
				img = Image.fromarray(np.uint8(img))
				img.save(save_folder+"/i_"+str(ep)+'_'+str(i)+".jpg")
			elif format.upper()=='NPZ':
				np.savez(save_folder+"/i_"+str(ep)+'_'+str(i)+".npz",img)

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

		x.save(self.folder+'/Inference_Result'+"/i"+str(num)+".jpg")
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

		x.save(self.folder+'/Inference_Result'+"/t"+str(num)+".jpg")
	def saveModel(self, model, name, num): #Save a Model
		json = model.to_json()
		with open(self.folder+'/Training_Weights'+"/"+name+".json", "w") as json_file:
			json_file.write(json)
		if name == 'discriminator_model':
			model.save_weights(self.folder+'/Training_Weights'+"/"+name+".hdf5")
		else:
			model.save_weights(self.folder+'/Training_Weights'+"/"+name+"_"+str(num)+".hdf5")
	def loadModel(self, name, num): #Load a Model
		#load_architecture
		json_file = open(self.folder+'/Training_Weights'+"/"+name+".json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json, custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
		if name == 'discriminator_model':
			model.load_weights(self.folder+'/Training_Weights'+"/"+name+".hdf5")
		else:
			model.load_weights(self.folder+'/Training_Weights'+"/"+name+"_"+str(num)+".hdf5")

		return model
	def save(self, num): #Save JSON and Weights
		self.saveModel(self.GAN.G, "generator_model", num)
		self.saveModel(self.GAN.D, "discriminator_model", num)
	def load(self, num, mode='TRAIN'): #Load JSON and Weights
		steps1 = self.GAN.steps

		self.GAN = None
		self.GAN = GAN(self.folder)

		#Load Models
		self.GAN.G = self.loadModel("generator_model", num)
		if mode not in ['POSTP','EVAL']:
			self.GAN.D = self.loadModel("discriminator_model", num)

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
		T=open(self.folder+'/Training_History/Histogram_Data.csv','w')
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
			print(']')
			if (i+1) % C.nb_evals == 0:
				#Save Model
				UL.save_model(self.GAN.G,'./'+self.folder+'/Training_Weights'+'/generator.json','./'+self.folder+'/Training_Weights'+'/generator_model_%03d.hdf5'% (i + 1))
				UL.save_model(self.GAN.D,'./'+self.folder+'/Training_Weights'+'/discriminator.json','./'+self.folder+'/Training_Weights'+'/discriminator_model.hdf5')
				T.write(str(i+1)+',')
				for p in range(0,len(a)):
					T.write(str(a[p])+',')
				for q in range(0,len(b)):
					T.write(str(b[q])+',')
				T.write('\n')
				#print('Step = ',i, ' Model Saved')
		T.close()

def Main():
	#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.85,allow_growth = True)
	#config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.compat.v1.Session(config=config)
	tf.compat.v1.keras.backend.set_session(session)

	parser = argparse.ArgumentParser(description='Synthetic Head Model Generation',
									usage='Generate artificial head phantoms using GAN network',
									epilog='Give proper arguments')
	parser.add_argument('-f',"--res_fold",metavar='',help="Save training and testing results in folder",default='Resultant_Folder')
	parser.add_argument('-m',"--mode",metavar='',help="Define Train mode or Inference Mode")
	parser.add_argument('-ep',"--epoch",metavar='',help="Define epoch number for post processing or evaluation",default=None)
	parser.add_argument('-s',"--sample",metavar='',help="Number of samples to evaluate through GAN generation",default=1000)

	parser.add_argument('-ne',"--num_evals",metavar='',help="How many steps between each inference",default=C.nb_evals)
	args = parser.parse_args()
	folder = args.res_fold
	mode = args.mode
	inference_num = int(args.epoch)
	infer_step = int(args.num_evals)
	samples = int(args.sample)

	#Check whether specific folders are present or not....if not create them
	FC.Folder_creation(folder)

	if mode.upper()=='TRAIN':
		#Train
		model = WGAN(folder)
		print(model.DisModel.summary())
		print(model.AdModel.summary())
		print(model.generator.summary())
		if C.resume_status:
			model.load(C.initial_iter)
		model.train()
	elif mode.upper()=='TEST':
		#Test
		weight_list=os.listdir(folder+'/Training_Weights')
		L=list()
		for i in weight_list:
			num=i.split('_')[-1]
			if num.split('.')[0] not in ['discriminator','model','generator']:
				L.append(int(num.split('.')[0]))
		init = min(L)
		L=max(L)
		print('\n\n')
		for i in range(init,L+1,infer_step):
			ep = '0'+str(i) if i < 100 else i
			model = WGAN(folder)
			print('Loaded Model = ',ep,end = "\r",flush=True)
			model.load(ep)
			model.evaluate(ep,folder+'/Inference_Result',format='jpg')
			K.clear_session()
	elif mode.upper() == 'POSTP':#Here the images the quantized
		init=inference_num
		end=init
		infer_step=1
		save_folder=folder+'/Analysis/Evaluating_Images'

		for i in range(init,end+1,infer_step):
			ep = '0'+str(i) if i < 100 else i
			model = WGAN(folder)
			print('Loaded Model = ',ep)
			model.load(ep,mode='POSTP')
			save_folder = save_folder +'/'+str(i)
			if not os.path.exists(save_folder):
				os.makedirs(save_folder)
			model.evaluate(ep,save_folder,num=1000,verbose=1,format='npz')
			K.clear_session()
	elif mode.upper() == 'EVAL':
		epochs = os.listdir(folder + '/Analysis/Evaluating_Images')
		epochs = [int(ep) for ep in epochs]
		epochs.sort()
		print(epochs)
		if inference_num not in [-1]:
			epochs = [epochs[inference_num]]

		epoch_count = 0
		file_count = 0
		for i in range(0,len(epochs)):
			ep = epochs[i]
			print('Sl = ',epoch_count,' Epoch = ',ep)
			print('Counting Area')
			GE.pixel_count(folder,samples,ep)
			epoch_count += 1
			file_count += 1
			#print('\n\n\n')
			'''
			print('Calculating IS')
			with open(folder+'/Analysis/IS_Stepwise.csv','a') as Q:
				is_avg, is_std = GE.calculate_inception_score(folder,samples,ep)
				Q.write(str(ep)+','+str(is_avg)+','+str(is_std)+'\n')
				K.clear_session()

			#print('Inception Score (Average) = ',is_avg)
			#print('Inception Score (STD) = ',is_std)
			
			print('Calculating FID')
			with open(folder+'/Analysis/FID_Stepwise.csv','a') as Q:
				FID_rf = GE.calculate_fid(folder,samples,ep)
				Q.write(str(ep)+','+str(FID_rf)+'\n')
				K.clear_session()
			#print('FID Score (real and fake) = ',FID_rf)
			'''
	else:
		print('Define either "Train" or "Test" or "Postp" or "Eval" in the command line argument as a -m option')

Main()

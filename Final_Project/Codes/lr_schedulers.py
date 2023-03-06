# mlp with cosine annealing learning rate schedule on blobs problem
from tensorflow import keras
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras import backend as K
from math import pi,cos,floor
import math
import numpy as np
import config
global C
C = config.Config()

class Constant_LR(callbacks.Callback):
	def __init__(self, initial_lrate=C.initial_lrate):
		self.initial_lrate = initial_lrate

	def constant_lr(self,epoch):
		lrate = self.initial_lrate
		return lrate

	def on_epoch_begin(self, epoch, logs=None):
		# calculate learning rate
		lr = self.constant_lr(epoch)
		# set learning rate
		K.set_value(self.model.optimizer.lr, lr)

class Step_Decay(callbacks.Callback):
	def __init__(self, drop=0.1,epochs_drop=5,initial_lrate=C.initial_lrate):
		self.initial_lrate = initial_lrate
		self.drop = drop
		self.epochs_drop = epochs_drop

	def step_decay(self,epoch):
		lrate = self.initial_lrate * math.pow(self.drop, math.floor((1+epoch)/self.epochs_drop))
		return lrate

	def on_epoch_begin(self, epoch, logs=None):
		# calculate learning rate
		lr = self.step_decay(epoch)
		# set learning rate
		K.set_value(self.model.optimizer.lr, lr)

class Exponential_Decay(callbacks.Callback):
	def __init__(self,initial_lrate=C.initial_lrate,drop=0.1):
		self.initial_lrate = initial_lrate
		self.drop = drop

	def exp_decay(self,epoch):
		lrate = self.initial_lrate * math.exp(-self.drop*epoch)
		return lrate

	def on_epoch_begin(self, epoch, logs=None):
		lr = self.exp_decay(epoch)
		K.set_value(self.model.optimizer.lr, lr)

class CosineAnnealingLearningRateSchedule(callbacks.Callback):
	# constructor
	def __init__(self,folder,number_of_epochs=C.number_of_epochs,cycle_length=50,lr_max=C.initial_lrate):
		self.epochs = number_of_epochs
		self.cycle_length = cycle_length
		self.cycles = self.epochs/self.cycle_length
		self.lr_max = lr_max
		self.folder = folder

	# calculate learning rate for an epoch
	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
		epochs_per_cycle = floor(n_epochs/n_cycles)
		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
		return lrate_max/2 * (cos(cos_inner) + 1)

	# calculate and set learning rate at the start of the epoch
	def on_epoch_begin(self, epoch, logs=None):
		# calculate learning rate
		lr = self.cosine_annealing(epoch,self.epochs,self.cycles,self.lr_max)
		# set learning rate
		K.set_value(self.model.optimizer.lr, lr)

	# save models at the end of each cycle
	def on_epoch_end(self, epoch, logs={}):
		# check if we can save model
		epochs_per_cycle = floor(self.epochs / self.cycles)
		if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
			# save model to file
			filename = folder+"/Ensemble_Models/Snapshot_Ensembling/model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
			self.model.save(filename)
			print( ' >saved snapshot %s, epoch %d ' % (filename, epoch))

class CyclicLR(callbacks.Callback):
	"""This callback implements a cyclical learning rate policy (CLR).
	The method cycles the learning rate between two boundaries with
	some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
	The amplitude of the cycle can be scaled on a per-iteration or
	per-cycle basis.
	This class has three built-in policies, as put forth in the paper.
	"triangular":
		A basic triangular cycle w/ no amplitude scaling.
	"triangular2":
		A basic triangular cycle that scales initial amplitude by half each cycle.
	"exp_range":
		A cycle that scales initial amplitude by gamma**(cycle iterations) at each
	cycle iteration.
	For more detail, please see paper.

	# Example
	```python
		clr = CyclicLR(base_lr=0.001, max_lr=0.006,
						step_size=2000., mode='triangular')
		model.fit(X_train, Y_train, callbacks=[clr])
	```

	Class also supports custom scaling functions:
	```python
		clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
		clr = CyclicLR(base_lr=0.001, max_lr=0.006,
							step_size=2000., scale_fn=clr_fn,
							scale_mode='cycle')
		model.fit(X_train, Y_train, callbacks=[clr])
	```
	# Arguments
		base_lr: initial learning rate which is the
			lower boundary in the cycle.
		max_lr: upper boundary in the cycle. Functionally,
			it defines the cycle amplitude (max_lr - base_lr).
			The lr at any cycle is the sum of base_lr
			and some scaling of the amplitude; therefore
			max_lr may not actually be reached depending on
			scaling function.
		step_size: number of training iterations per
			half cycle. Authors suggest setting step_size
			2-8 x training iterations in epoch.
		mode: one of {triangular, triangular2, exp_range}.
			Default 'triangular'.
			Values correspond to policies detailed above.
			If scale_fn is not None, this argument is ignored.
		gamma: constant in 'exp_range' scaling function:
			gamma**(cycle iterations)
		scale_fn: Custom scaling policy defined by a single
			argument lambda function, where
			0 <= scale_fn(x) <= 1 for all x >= 0.
			mode paramater is ignored
		scale_mode: {'cycle', 'iterations'}.
			Defines whether scale_fn is evaluated on
			cycle number or cycle iterations (training
			iterations since start of cycle). Default is 'cycle'.
	"""
	def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
				gamma=1., scale_fn=None, scale_mode='cycle'):
		super(CyclicLR, self).__init__()

		self.base_lr = base_lr
		self.max_lr = max_lr
		self.step_size = step_size
		self.mode = mode
		self.gamma = gamma
		if scale_fn == None:
			if self.mode == 'triangular':
				self.scale_fn = lambda x: 1.
				self.scale_mode = 'cycle'
			elif self.mode == 'triangular2':
				self.scale_fn = lambda x: 1/(2.**(x-1))
				self.scale_mode = 'cycle'
			elif self.mode == 'exp_range':
				self.scale_fn = lambda x: gamma**(x)
				self.scale_mode = 'iterations'
		else:
			self.scale_fn = scale_fn
			self.scale_mode = scale_mode
		self.clr_iterations = 0.
		self.trn_iterations = 0.
		self.history = {}

		self._reset()

	def _reset(self, new_base_lr=None, new_max_lr=None,new_step_size=None):
		if new_base_lr != None:
			self.base_lr = new_base_lr
		if new_max_lr != None:
			self.max_lr = new_max_lr
		if new_step_size != None:
			self.step_size = new_step_size
		self.clr_iterations = 0.

	def clr(self):
		cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
		x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
		if self.scale_mode == 'cycle':
			return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
		else:
			return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

	def on_train_begin(self, logs={}):
		logs = logs or {}
		if self.clr_iterations == 0:
			K.set_value(self.model.optimizer.lr, self.base_lr)
		else:
			K.set_value(self.model.optimizer.lr, self.clr())

	def on_batch_end(self, epoch, logs=None):
		logs = logs or {}
		self.trn_iterations += 1
		self.clr_iterations += 1

		self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
		self.history.setdefault('iterations', []).append(self.trn_iterations)

		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)

		K.set_value(self.model.optimizer.lr, self.clr())

def Learning_Rate_Scheduler(folder,lr_scheduler='constant',**kwargs):
	LR_scheduler = {
		'constant':Constant_LR(**kwargs),
		'step':Step_Decay(**kwargs),
		'exponential':Exponential_Decay(**kwargs),
		'cosine_annealing':CosineAnnealingLearningRateSchedule(folder,**kwargs),
		'cyclic_triangular':CyclicLR(**kwargs),
		'cyclic_triangular_variable_height':CyclicLR(**kwargs),
		'cyclic_triangular_exponential_height':CyclicLR(**kwargs)
	}
	return LR_scheduler[lr_scheduler]

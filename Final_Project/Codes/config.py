
#for i in $(sudo lsof /dev/nvidia0 | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done
class Config:
	def __init__(self):
		self.discriminator_loss_type='binary_crossentropy'
		self.gan_loss_type='binary_crossentropy'
		self.discriminator_optimizer='adam'
		self.gan_optimizer='adam'
		self.discriminator_activation_function = 'sigmoid'
		self.generator_activation_function='sigmoid'
		self.metrics=['accuracy']
		self.train_3D=False
		self.single_train_sample_mode=False#For 3D data load each sample in a batch (one at a time) to the GPU memory
		self.inference_epoch=''
		self.latent_dimension=50
		self.conditional_GAN=False
		self.conditional_classes=2
		self.full_database_load = False
		if self.full_database_load:
			self.queue_run = False
		else:
			self.queue_run = True
		self.nb_queues=10
		self.len_queue=6
		self.nb_epochs=10000
		self.post_process_step=250000
		self.nb_batches=32#During the training of discriminaor half-batch is used at a time for real and fake data
		self.nb_evals=50#number of steps between GAN evaluations
		self.nb_example=100#number of examples to generate during evaluation
		self.early_stopage_patience = 50
		self.lr_plateau_patience = 20
		self.nb_segments=[0,1,2,3,4,5]#for single tissue type [0,tissue number]; for multiple tissue types [0,1,2,3,4,5] etc
		self.tissue_map={'1':'WM','2':'GM','3':'CSF','4':'Skull','5':'Scalp','6':'Eyes'}
		self.image_shape=[256,256]
		self.image_channels=[1]#Declare it inside a list for the sake of code
		self.filter_scale=100#Percentage of original filters that you want to use. values are in range (1-100)%. Reduce it for 3D data or for low memory machines/ GPUs
		self.channel_pos=-1#-1 for tensorflow training (channel last), 0 for pytorch training (channel first)
		self.number_of_gpus=1
		self.Train_as_SeedGAN = False
		self.label_smoothing=False
		self.noisy_labels=False
		self.noise_p=0.05;

		#learning rate Scheduler
		self.scheduler = 'constant'
		self.initial_lrate=0.00002*self.number_of_gpus
		self.lr_scheduler_parameters = {
			'constant':{'lr':self.initial_lrate},
			'step':{'drop':0.1,'epochs_drop':5,'initial_lrate':self.initial_lrate},
			'exponential':{'initial_lrate':self.initial_lrate,'drop':0.1},
			'cosine_annealing':{'number_of_epochs':self.nb_epochs,'cycle_length':50,'lr_max':self.initial_lrate},
			'cyclic_triangular':{'base_lr':0.001, 'max_lr':0.01, 'step_size':2000., 'mode':'triangular',
													'gamma':1., 'scale_fn':None, 'scale_mode':'cycle'},
			'cyclic_triangular_variable_height':{'base_lr':0.001, 'max_lr':0.01, 'step_size':2000., 'mode':'triangular2',
													'gamma':1., 'scale_fn':None, 'scale_mode':'cycle'},
			'cyclic_triangular_exponential_height':{'base_lr':0.001, 'max_lr':0.01, 'step_size':2000., 'mode':'exp_range',
													'gamma':1., 'scale_fn':None, 'scale_mode':'cycle'}
		}
		self.classes=2#number of classes for the discriminator
		self.data_list= 'subject_list_'+str(self.image_shape[0])+'_mri2mesh.csv'#a list of all data
		self.data_directory='/scratch/bell/hasan34/data/head_model_data/headreco_models/Voxelized_Heads_Headreco/'+str(self.image_shape[0])+'_'+str(self.image_shape[1])#data directory
		self.shuffle = True#Data shuffle during training
		self.resume_status = True
		self.initial_iter = 56050#Use it for model Training Resuming;Initially it is zero,

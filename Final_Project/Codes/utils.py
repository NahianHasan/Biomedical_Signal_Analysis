from numpy import ones,zeros
from numpy.random import rand,randint,randn,random,choice
from tensorflow.keras.models import model_from_json
from numpy import expand_dims
import os,glob
import numpy as np
import scipy.io
import time
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import config
global C
C=config.Config()

def prepare_subject_list(directory):
	if not os.path.exists('subject_list_'+str(C.image_shape[0])+'.csv'):
		G=open('subject_list_'+str(C.image_shape[0])+'.csv','w')
		subjects=os.listdir(directory)
		for S in subjects:
			if os.path.exists(directory+'/'+S+'/voxelized_head_'+str(C.image_shape[0])+'.mat'):
				G.write(directory+'/'+S+'/voxelized_head_'+str(C.image_shape[0])+'.mat\n')
			else:

				continue
		G.close()
	else:
		print('\n\nSubject List Is Already Prepared\n\n')

def load_real_samples(selected_list,channel):
	X=list()
	for S in selected_list:
		H=scipy.io.loadmat(S)
		S = H['voxelRegions']
		if C.train_3D == False:
			S = S[:,int(C.image_shape[0]/2),:]
		if len(C.nb_segments) > 2:
			#For now remove eyes
			S[np.where(S == 6)] = 0
		else:
			#print(S.shape)
			S[np.where(S != max(C.nb_segments))] = 0#make all tissues zero except the desired one
			S[np.where(S != 0)] = 1#get the binary image
		X.append(S)
		#print(np.unique(S))

	# convert from unsigned ints to floats
	X = np.asarray(X)
	X = expand_dims(X, axis=channel)
	X = X.astype('float32')
	#X is a list of voxelized heads. each head has unique values from 0-5.
	# scale from [0,5] to [-1,1]
	if C.generator_activation_function=='tanh':
		if len(C.nb_segments) > 2:
			X = (X - (max(X)/2.0)) / (max(X)/2.0)
		else:
			X[np.where(X == 0)] = -1
			#print(np.unique(X))
	elif C.generator_activation_function=='sigmoid':
		if len(C.nb_segments) > 2:
			X = X / np.max(X)
		else:
			X = X
	return X


# load and prepare cifar10 training images
def load_records(subject_list):
	record_list=list()
	with open(subject_list,'r') as F:
		line=F.readline()
		while(line):
			record_list.append(line.split(',')[0][0:-1])
			line=F.readline()
	if C.full_database_load:
		record_list = load_real_samples(record_list,C.channel_pos)

	return record_list

# example of smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
	return y - 0.3 + (random(y.shape) * 0.5)

# example of smoothing class=0 to [0.0, 0.3]
def smooth_negative_labels(y):
	return y + random(y.shape) * 0.3

# randomly flip some labels
def noisy_labels(y, p_flip):
	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0])
	# choose labels to flip
	flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]
	return y

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = list(randint(0, len(dataset), n_samples))
	# retrieve selected image list
	X_list = [dataset[i] for i in ix]
	if not C.full_database_load:
		X=load_real_samples(X_list, C.channel_pos)
	else:
		X = np.array(X_list)
	# generate ✬ real ✬ class labels (1)
	Y = ones((n_samples, 1))
	if C.noisy_labels:
		noisy_labels(Y, C.noise_p)
	if C.label_smoothing:
		Y = smooth_positive_labels(Y)
	return X, Y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	X = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	X = X.reshape(n_samples, latent_dim)
	return X

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	X = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(X)
	# create ✬ fake ✬ class labels (0)
	Y = zeros((n_samples, 1))
	if C.noisy_labels:
		noisy_labels(Y, C.noise_p)
	if C.label_smoothing:
		Y = smooth_negative_labels(Y)
	return X, Y

# create and save a plot of generated data
def save_result(folder,examples,ep):
	save_dir=folder+'/Inference_Result/'
	if C.generator_activation_function=='tanh':
		examples=(examples + 1) / 2.0
	elif C.generator_activation_function=='sigmoid':
		examples=examples*255
	for i in range(10 * 10):
		# define subplot
		plt.subplot(10, 10, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i,:,:,0], cmap='gray_r')
		# save plot to file
	plt.savefig(save_dir+'generated_plot_'+str(ep)+'.png')
	plt.close()

# create a line plot of loss for the gan and save to file
def plot_history(folder):
	epoch, d_loss_1_hist, d_loss_2_hist, gan_loss_hist, d_acc_1_hist, d_acc_2_hist = list(), list(), list(), list(), list(), list()
	T=open(folder+'/Training_History/Histogram_Data.csv','r')
	line = T.readline()
	line = T.readline()
	while(line):
		epoch.append(line.split(',')[0])
		d_loss_1_hist.append(line.split(',')[1])
		d_loss_2_hist.append(line.split(',')[2])
		gan_loss_hist.append(line.split(',')[3])
		d_acc_1_hist.append(line.split(',')[4])
		d_acc_2_hist.append(line.split(',')[5])
		line=T.readline()

	plt.subplot(2, 1, 1)
	plt.plot(d_loss_1_hist, label='d-loss-real')
	plt.plot(d_loss_2_hist, label='d-loss-fake')
	plt.plot(gan_loss_hist, label='gan-loss')
	plt.legend()
	# plot discriminator accuracy
	plt.subplot(2, 1, 2)
	plt.plot(d_acc_1_hist, label='d-acc-real')
	plt.plot(d_acc_2_hist, label='d-acc-fake')
	plt.legend()
	# save plot to file
	plt.savefig(folder+'/Training_History/plot_line_plot_loss.png')
	plt.close()
	T.close()
	print('Done Plotting, You can close the program now')

def save_model(model, json_path, weight_path):
	model_json = model.to_json()
	with open(json_path, "w") as json_file:
		json_file.write(model_json)
	model.save(weight_path)

# evaluate the discriminator and plot real and fake points
def summarize_performance(folder, epoch, generator, discriminator, gan, latent_dim, n=C.nb_example):
	# save the generator model tile file
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	save_result(folder,x_fake,epoch+1)
	save_model(generator,folder+'/Model_Figures/generator.json',folder+'/Weights/generator_model_%03d.hdf5'% (epoch + 1))
	save_model(discriminator,folder+'/Model_Figures/discriminator.json',folder+'/Weights/discriminator_model_%03d.hdf5'% (epoch + 1))
	save_model(gan,folder+'/Model_Figures/gan.json',folder+'/Weights/gan_model_%03d.hdf5'% (epoch + 1))

def load_model(json_file,weight_folder,mode,epoch_number):
	#load_architecture
	json_file = open(json_file,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	#Load weights
	if mode=='generator':
		model_weight_file =  weight_folder+'/generator_model_'+str(epoch_number)+'.hdf5'
	elif mode=='discriminator':
		model_weight_file =  weight_folder+'/discriminator_model_'+str(epoch_number)+'.hdf5'
	print('\n\n\n', model_weight_file, '\n\n\n')
	model.load_weights(model_weight_file)

	return model

# Transform train_on_batch return value
# to dict expected by on_batch_end callback
def named_logs(metrics, logs):
	result = {}
	for l in range(0,len(metrics)):
		result[metrics[l]] = logs[l]
	return result

def compare_images(start,end,step,folder):
	original_image=scipy.io.loadmat('')
	subp_count=1
	for i in range(start,end+1,step):
		im = np.load(folder+'/i'+str(i)+'.npz',allow_pickle=True)
		img = im['arr_0']
		plt.subplot(221)
		plt.imshow(img)
		plt.title('Generated Image')
		plt.axis('off')
		plt.tight_layout()

		plt.subplot(224)
		F=np.reshape(img,[img.shape[0],img.shape[1]])
		plt.hist(F,bins=256)

		hist=np.histogram(img,bins=6)
		pairs=sorted(zip(hist[0],hist[1]))
		tuples=zip(*pairs)
		list1,list2=[list(tuple) for tuple in tuples]
		list2.sort()
		print(list2)

		img=np.array(img)
		img[np.where(img < list2[-6]+(list2[-5]-list2[-6])/2)] = 0
		img[np.where((img >= list2[-6]+(list2[-5]-list2[-6])/2) & (img < list2[-5]+(list2[-4]-list2[-5])/2))] = list2[-5]
		img[np.where((img >= list2[-5]+(list2[-4]-list2[-5])/2) & (img < list2[-4]+(list2[-3]-list2[-4])/2))] = list2[-4]
		img[np.where((img >= list2[-4]+(list2[-3]-list2[-4])/2) & (img < list2[-3]+(list2[-2]-list2[-3])/2))] = list2[-3]
		img[np.where((img >= list2[-3]+(list2[-2]-list2[-3])/2) & (img < list2[-2]+(list2[-1]-list2[-2])/2))] = list2[-2]
		img[np.where(img >= list2[-2]+(list2[-1]-list2[-2])/2)] = 255
		plt.subplot(222)
		imgplot = plt.imshow(img)
		plt.axis('off')
		plt.tight_layout()
		plt.title('Quantized Image')


		plt.subplot(223)
		voxel=original_image['voxelRegions']
		voxel = voxel[:,64,:]
		voxel[np.where(voxel == 6)] = 0
		img_org=voxel*51
		plt.imshow(img_org)
		plt.title('Real Sample Head Voxels')
		#plt.show()
		#time.sleep(5)
		plt.savefig("./Results_mri2mesh_256/i"+str(i)+".jpg")


		uniq_vals=np.unique(img_org)
		print(uniq_vals)
		nz=5
		img_org[np.where(img_org != uniq_vals[nz-6])] = 0
		img_org[np.where(img_org == uniq_vals[nz-6])] = 255
		plt.subplot(223)
		plt.imshow(img_org,cmap=plt.cm.gray)
		img_org=np.reshape(img_org,[img_org.shape[0], img_org.shape[1]])
		open_img = ndimage.binary_opening(img_org, structure=np.ones((1,1))).astype(np.int)
		#img[np.where(img != 255 )] = 0
		#plt.subplot(130+subp_count+2)
		plt.subplot(224)
		plt.imshow(open_img,cmap=plt.cm.gray)
		plt.axis('off')


		uniq_vals=np.unique(img)
		print(uniq_vals)
		img[np.where(img != uniq_vals[nz-6])] = 0
		img[np.where(img == uniq_vals[nz-6])] = 255
		plt.subplot(221)
		plt.imshow(img,cmap=plt.cm.gray)
		#img = ndimage.grey_dilation(img, size=(5, 5), structure=np.ones((5, 5)))
		#img = ndimage.binary_dilation(img)
		img=np.reshape(img,[img.shape[0], img.shape[1]])
		open_img = ndimage.binary_opening(img, structure=np.ones((2,2))).astype(np.int)

		#img[np.where(img != 255 )] = 0
		#plt.subplot(130+subp_count+2)
		plt.subplot(222)
		plt.imshow(open_img,cmap=plt.cm.gray)
		plt.axis('off')
		#plt.show()
		#time.sleep(5)
		plt.savefig("./Results_mri2mesh_256/WM_i"+str(i)+".jpg")


		#plt.title('Quantized Image')

	'''
	plt.subplot(223)
	voxel=original_image['voxelRegions']
	voxel = voxel[:,64,:]
	voxel[np.where(voxel == 6)] = 0
	print(np.unique(voxel))
	plt.imshow(voxel*51)
	plt.axis('off')
	plt.title('Original voxelized head')
	plt.show()
	'''

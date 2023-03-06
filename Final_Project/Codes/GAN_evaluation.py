from math import floor
import utils as UL
import os,glob
from numpy import ones,expand_dims,log,mean,std,cov,trace,exp,iscomplexobj
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
import numpy as np
from numpy.random import random
from scipy.linalg import sqrtm
import config
import tensorflow as tf
global C
C = config.Config()

# assumes images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(folder,N, ep, n_split=100, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# convert from uint8 to float32
	analysis_dir=folder+'/Analysis/Evaluating_Images'
	image_list=list()
	image_list = list()
	n_part = floor(N / n_split)
	scores = list()
	for i in range(N):
		im = np.load(analysis_dir+'/'+str(ep)+'/i_'+str(ep)+'_'+str(i)+'.npz',allow_pickle=True)
		img = im['arr_0']
		image_list.append(img)
		if len(image_list) == n_part:
			image_list = np.asarray(image_list)
			processed = image_list.astype('float32')
			# pre-process raw images for inception v3 model
			processed = preprocess_input(processed)
			#reshape imags by padding with zeros and
			processed = np.pad(processed, ((0,0),(floor((299-processed.shape[1])/2),floor((299-processed.shape[1])/2)),(floor((299-processed.shape[1])/2),floor((299-processed.shape[1])/2)),(0,0)), mode='constant', constant_values=0)
			processed = np.pad(processed, ((0,0),(0,1),(0,1),(0,0)), mode='constant', constant_values=0)
			processed = np.concatenate((processed,processed,processed),axis=-1)
			# predict class probabilities for images
			yhat = model.predict(processed)
			# retrieve p(y|x)
			p_yx = yhat
			# calculate p(y)
			p_y = expand_dims(p_yx.mean(axis=0), 0)
			# calculate KL divergence using log probabilities
			kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
			# sum over classes
			sum_kl_d = kl_d.sum(axis=1)
			# average over images
			avg_kl_d = mean(sum_kl_d)
			# undo the log
			is_score = exp(avg_kl_d)
			# store
			scores.append(is_score)
			# average across images
			image_list = list()
			print(is_score)
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std

# calculate frechet inception distance
def calculate_fid(folder,N,ep):
	model = InceptionV3(include_top=False, pooling= 'avg' , input_shape=(299,299,3))
	dataset = UL.load_records(C.data_list)
	#analysis_dir=folder+'/Analysis/Evaluating_Images_Step_'+str(C.post_process_step)
	analysis_dir=folder+'/Analysis/Evaluating_Images'
	im_f = list()

	print('\n\n')
	for i in range(N):
		im = np.load(analysis_dir+'/'+str(ep)+'/i_'+str(ep)+'_'+str(i)+'.npz',allow_pickle=True)
		img = im['arr_0']
		im_f.append(img)
		print('Number of Image Checked = ',i,end = "\r",flush=True)
		if len(im_f) == 100:
			im_r,_ = UL.generate_real_samples(dataset,100)
			im_r = im_r * 255
			im_f = np.asarray(im_f)

			'''
			for T in [4,5]:
				im_r[np.where(im_r == T*51)] = 0
				im_f[np.where(im_f == T*51)] = 0
			'''
			im_r,im_f = im_r.astype('float32'),im_f.astype('float32')
			# pre-process raw images for inception v3 model
			im_r,im_f = preprocess_input(im_r), preprocess_input(im_f)
			#reshape imags by padding with zeros and
			im_r = np.pad(im_r, ((0,0),(floor((299-im_r.shape[1])/2),floor((299-im_r.shape[1])/2)),(floor((299-im_r.shape[1])/2),floor((299-im_r.shape[1])/2)),(0,0)), mode='constant', constant_values=0)
			im_r = np.pad(im_r, ((0,0),(0,1),(0,1),(0,0)), mode='constant', constant_values=0)
			im_r = np.concatenate((im_r,im_r,im_r),axis=-1)

			im_f = np.pad(im_f, ((0,0),(floor((299-im_f.shape[1])/2),floor((299-im_f.shape[1])/2)),(floor((299-im_f.shape[1])/2),floor((299-im_f.shape[1])/2)),(0,0)), mode='constant', constant_values=0)
			im_f = np.pad(im_f, ((0,0),(0,1),(0,1),(0,0)), mode='constant', constant_values=0)
			im_f = np.concatenate((im_f,im_f,im_f),axis=-1)

			if i <= 100:
				act_r = model.predict(im_r)
				act_f = model.predict(im_f)
			else:
				act_r = np.concatenate((act_r,model.predict(im_r)),axis=0)
				act_f = np.concatenate((act_f,model.predict(im_f)),axis=0)
			im_f = list()

	print('\n\n')
	##FID between Real and Fake
	# calculate mean and covariance statistics
	mu1, sigma1 = act_r.mean(axis=0), cov(act_r,rowvar=False)
	mu2, sigma2 = act_f.mean(axis=0), cov(act_f,rowvar=False)
	# calculate sum squared difference between
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

#calculate pixel statistics
def pixel_count(folder,N,ep):
	dataset = UL.load_records(C.data_list)
	real_pixel_count = {'1':[],'2':[],'3':[],'4':[],'5':[]}
	fake_pixel_count = {'1':[],'2':[],'3':[],'4':[],'5':[]}
	analysis_dir=folder+'/Analysis/Evaluating_Images'
	im_f = list()

	print('\n\n')
	for i in range(N):
		im = np.load(analysis_dir+'/'+str(ep)+'/i_'+str(ep)+'_'+str(i)+'_processed.npz',allow_pickle=True)
		img = im['arr_0']
		for j in range(1,6):
			fake_pixel_count[str(j)].append(np.shape(np.where(img == j))[1])
	np.savez(analysis_dir+'/'+str(ep)+'/fake_pixel_count.npz',fake_pixel_count)

	if not os.path.exists('real_pixel_count.npz'):
		for i in range(len(dataset)):
			im_r = UL.load_real_samples([dataset[i]],C.channel_pos)
			im_r = im_r * 255
			im_r = im_r.astype('float32')
			for j in range(1,6):
				real_pixel_count[str(j)].append(np.shape(np.where(im_r[0] == j*51))[1])
		np.savez('real_pixel_count.npz',real_pixel_count)
	'''
	else:
		real_data = np.load('real_pixel_count.npz',allow_pickle=True)
		real_pixel_count = real_data['arr_0']
	'''
	#data_fake = [fake_pixel_count[str(1)],fake_pixel_count[str(2)],fake_pixel_count[str(3)],fake_pixel_count[str(4)],fake_pixel_count[str(5)]]
	#data_real = [real_pixel_count[str(1)],real_pixel_count[str(2)],real_pixel_count[str(3)],real_pixel_count[str(4)],real_pixel_count[str(5)]]
	'''
	for T in range(1,6):
		data = [fake_pixel_count[str(T)],real_pixel_count[str(T)]]
		fig, ax1 = plt.subplots(figsize=(12, 8))
		fig.canvas.set_window_title('Pixel Statistics of Real and Generated Head Models')
		plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
		bp = plt.boxplot(data,notch=0, sym='+', vert=1, whis=1.5)
		plt.setp(bp['boxes'], color='black')
		plt.setp(bp['whiskers'], color='black')
		plt.setp(bp['fliers'], color='red', marker='+')
		ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
		ax1.set_axisbelow(True)
		ax1.set_title('Pixel Statistics of Real and Generated Head Models', fontsize=15)
		ax1.set_xlabel('Tissue Type', fontsize=15)
		ax1.set_ylabel('Pixel Count', fontsize=15)
		# Now fill the boxes with desired colors
		boxColors = ['darkkhaki', 'royalblue']
		numBoxes = 2
		medians = list(range(numBoxes))
		for i in range(numBoxes):
			box = bp['boxes'][i]
			boxX = []
			boxY = []
			for j in range(5):
				boxX.append(box.get_xdata()[j])
				boxY.append(box.get_ydata()[j])
			boxCoords = list(zip(boxX, boxY))
			# Alternate between Dark Khaki and Royal Blue
			k = i % 2
			boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
			ax1.add_patch(boxPolygon)
			# Now draw the median lines back over what we just filled in
			med = bp['medians'][i]
			medianX = []
			medianY = []
			for j in range(2):
				medianX.append(med.get_xdata()[j])
				medianY.append(med.get_ydata()[j])
				plt.plot(medianX, medianY, 'k')
				medians[i] = medianY[0]
			# Finally, overplot the sample averages, with horizontal alignment
			# in the center of each box
			plt.plot([np.average(med.get_xdata())], [np.average(data[i])], color='w', marker='*', markeredgecolor='k')
		# Set the axes ranges and axes labels
		ax1.set_xlim(0.5, numBoxes + 0.5)
		top = max(max(data[0],data[1]))+2000
		bottom = 0
		ax1.set_ylim(bottom, top)
		#xtickNames = plt.setp(ax1, xticklabels=[C.tissue_map[str(j)] for j in range(1,6)])
		xtickNames = plt.setp(ax1, xticklabels=['Generated_'+C.tissue_map[str(T)],'Real_'+C.tissue_map[str(T)]])
		plt.setp(xtickNames, rotation=45, fontsize=15)
		# Due to the Y-axis scale being different across samples, it can be
		# hard to compare differences in medians across the samples. Add upper
		# X-axis tick labels with the sample medians to aid in comparison
		# (just use two decimal places of precision)
		pos = np.arange(numBoxes) + 1
		upperLabels = ['median = '+str(np.round(s, 2)) for s in medians]
		weights = ['bold', 'semibold']
		for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
			k = tick % 2
			ax1.text(pos[tick], top - (top*0.05), upperLabels[tick], horizontalalignment='center', fontsize=15, weight=weights[k], color=boxColors[k])
		#plt.show()
		savedir = analysis_dir+'/Step_'+str(ep)
		if not os.path.exists(savedir):
			os.makedirs(savedir)
		plt.savefig(savedir+'/'+C.tissue_map[str(T)]+'.png')
	'''

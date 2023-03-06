from math import floor
import utils as UL
import os,glob
from numpy import ones,expand_dims,log,mean,std,cov,trace,exp,iscomplexobj
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from numpy.random import random
from scipy.linalg import sqrtm
import config
global C
C = config.Config()



#calculate pixel statistics
def pixel_count(folder,N,epochs):
	data = {}
	for T in range(1,6):
		data[str(T)] = list()
	for q in range(0,len(epochs)):
		try:
			ep = epochs[q]
			analysis_dir=folder+'/Analysis/Evaluating_Images'
			fake_pixel_count = np.load(analysis_dir+'/'+str(ep)+'/fake_pixel_count.npz',allow_pickle=True)
			real_pixel_count = np.load('real_pixel_count.npz',allow_pickle=True)
			fake_pixel_count = fake_pixel_count['arr_0']
			real_pixel_count = real_pixel_count['arr_0']

			
			#data_fake = [fake_pixel_count[str(1)],fake_pixel_count[str(2)],fake_pixel_count[str(3)],fake_pixel_count[str(4)],fake_pixel_count[str(5)]]
			#data_real = [real_pixel_count[str(1)],real_pixel_count[str(2)],real_pixel_count[str(3)],real_pixel_count[str(4)],real_pixel_count[str(5)]]
			
			for T in range(1,6):
				data[str(T)].append(fake_pixel_count.item()[str(T)])
		except:
			print(q,' : Failed Epoch = ',ep)
			continue
	for T in range(1,6):
		fig, ax1 = plt.subplots(figsize=(12, 8))
		fig.canvas.set_window_title('Pixel Statistics of Real and Generated Head Models')
		plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
		bp = plt.boxplot(data[str(T)],notch=0, sym='+', vert=1, whis=1.5)
		plt.setp(bp['boxes'], color='black')
		plt.setp(bp['whiskers'], color='black')
		plt.setp(bp['fliers'], color='red', marker='+')
		ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
		ax1.set_axisbelow(True)
		ax1.set_title('Pixel Statistics of Real and Generated Head Models', fontsize=15)
		ax1.set_xlabel('Iteration', fontsize=15)
		ax1.set_ylabel('Area (unit=pixels)', fontsize=15)
		# Now fill the boxes with desired colors
		#boxColors = ['darkkhaki', 'royalblue']
		numBoxes = len(data[str(T)])
		medians = list(range(numBoxes))
		'''
		
		
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
			#plt.plot([np.average(med.get_xdata())], [np.average(data[str(T)][i])], color='w', marker='*', markeredgecolor='k')
		'''		
		# Set the axes ranges and axes labels
		ax1.set_xlim(0.5, numBoxes + 0.5)
		top = max(max(data[str(T)][0],data[str(T)][1]))+2000
		bottom = 0
		ax1.set_ylim(bottom, top)
		#xtickNames = plt.setp(ax1, xticklabels=[C.tissue_map[str(j)] for j in range(1,6)])
		xtickNames = plt.setp(ax1, xticklabels=[str(epochs[i]) for i in range(0,len(data[str(T)]))])
		plt.setp(xtickNames, rotation=90, fontsize=10, weight='bold')
		# Due to the Y-axis scale being different across samples, it can be
		# hard to compare differences in medians across the samples. Add upper
		# X-axis tick labels with the sample medians to aid in comparison
		# (just use two decimal places of precision)
		pos = np.arange(numBoxes) + 1
		upperLabels = ['median = '+str(np.round(s, 2)) for s in medians]
		'''
		weights = ['bold', 'semibold']
		for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
			k = tick % 2
			ax1.text(pos[tick], top - (top*0.05), upperLabels[tick], horizontalalignment='center', fontsize=15, weight=weights[k])
		#plt.show()
		'''
		plt.savefig('Pixel_Count_'+C.tissue_map[str(T)]+'.png')

folder = 'Results_mri2mesh_256_updated_all_tissues'
epochs = os.listdir(folder + '/Analysis/Evaluating_Images')
epochs = [int(ep) for ep in epochs]
epochs.sort()
pixel_count(folder,1000,epochs[5:352:5])

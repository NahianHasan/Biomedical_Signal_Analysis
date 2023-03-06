### This script post processes the images after the image quantization

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.axes_grid1 import ImageGrid

def image_morph(image,iter=1):
	kernel = np.ones((2,2), np.uint8)
	im_erode = cv2.erode(image, kernel, iterations=1)
	im_dilation = cv2.dilate(im_erode, kernel, iterations=1)
	if iter > 1:
		kernel = np.ones((3,3), np.uint8)
		im_erode2 = cv2.erode(im_dilation, kernel, iterations=1)
		im_dilation2 = cv2.dilate(im_erode2, kernel, iterations=2)
		return im_dilation,im_dilation2
	else:
		return im_dilation

def process_generated_image(file_path,save_path):
	if not os.path.exists(save_path+'.npz'):
		image = np.load(file_path,allow_pickle=True)
		image = image['arr_0']
		image = np.reshape(image,[image.shape[1],image.shape[2],image.shape[0]])
		WM = np.where(image==51, 1.0, 0.0)
		GM = np.where(image==102, 1.0, 0.0)
		CSF = np.where(image==153, 1.0, 0.0)
		Skull = np.where(image==204, 1.0, 0.0)
		Scalp = np.where(image==255, 1.0, 0.0)

		#Process WM
		WM_dialation,WM_dialation2 = image_morph(WM,iter=2)

		#Process GM
		kernel = np.ones((3,3), np.uint8)
		GM_dialation = cv2.dilate(GM, kernel, iterations=4)
		GM_dialation = GM_dialation.astype(np.uint8)

		#Find GM largest contour
		GM_contours,hierarchy = cv2.findContours(GM_dialation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour_lengths = []
		for i in range(0,len(GM_contours)):
			contour_lengths.append(len(GM_contours[i]))
		max_index = contour_lengths.index(max(contour_lengths))
		#Generate the GM mask by filling all the pixels inside the mask
		img_contours = np.zeros((GM.shape[0],GM.shape[1]))
		contour_GM = cv2.drawContours(img_contours, [GM_contours[max_index]],0, (2), thickness=1)
		img_contours = np.zeros((GM.shape[0],GM.shape[1]))
		mask_GM = cv2.drawContours(img_contours, [GM_contours[max_index]], 0, (2), thickness = -1)
		#Differentiate GM and WM in the same image
		segmented_GM_WM = mask_GM-WM_dialation2
		segmented_GM_WM = np.where(segmented_GM_WM==-1, 2, segmented_GM_WM)#Consider the small overlap positions as GM


		#Process CSF
		kernel = np.ones((3,3), np.uint8)
		CSF_dialation = cv2.dilate(CSF, kernel, iterations=4)
		CSF_dialation = CSF_dialation.astype(np.uint8)
		#Find CSF largest contour
		CSF_contours,hierarchy = cv2.findContours(CSF_dialation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour_lengths = []
		for i in range(0,len(CSF_contours)):
			contour_lengths.append(len(CSF_contours[i]))
		max_index = contour_lengths.index(max(contour_lengths))
		#Generate the GM mask by filling all the pixels inside the mask
		img_contours = np.zeros((CSF.shape[0],CSF.shape[1]))
		contour_CSF = cv2.drawContours(img_contours, [CSF_contours[max_index]], 0, (3), thickness = 1)
		img_contours = np.zeros((CSF.shape[0],CSF.shape[1]))
		mask_CSF = cv2.drawContours(img_contours, [CSF_contours[max_index]], 0, (3), thickness = -1)
		segmented_GM_WM_CSF = mask_CSF - segmented_GM_WM
		segmented_GM_WM_CSF = np.where(segmented_GM_WM_CSF<0, 3, segmented_GM_WM_CSF)#Consider the small overlap positions as CSF
		#Re-labelling tissues
		segmented_GM_WM_CSF += 1
		segmented_GM_WM_CSF = np.where(segmented_GM_WM_CSF==1, 0 , segmented_GM_WM_CSF)
		segmented_GM_WM_CSF = np.where(segmented_GM_WM_CSF==3, 1 , segmented_GM_WM_CSF)
		segmented_GM_WM_CSF = np.where(segmented_GM_WM_CSF==4, 3 , segmented_GM_WM_CSF)

		#Process Skull
		kernel = np.ones((3,3), np.uint8)
		Skull_dialation = cv2.dilate(Skull, kernel, iterations=3)
		Skull_dialation = Skull_dialation.astype(np.uint8)
		#Find Skull largest contour
		Skull_contours,hierarchy = cv2.findContours(Skull_dialation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour_lengths = []
		for i in range(0,len(Skull_contours)):
			contour_lengths.append(len(Skull_contours[i]))
		max_index = contour_lengths.index(max(contour_lengths))
		#Generate the GM mask by filling all the pixels inside the mask
		img_contours = np.zeros((Skull.shape[0],Skull.shape[1]))
		contour_Skull = cv2.drawContours(img_contours, [Skull_contours[max_index]], 0, (4), thickness = 1)
		img_contours = np.zeros((Skull.shape[0],Skull.shape[1]))
		mask_Skull = cv2.drawContours(img_contours, [Skull_contours[max_index]], 0, (4), thickness = -1)
		segmented_GM_WM_CSF_Skull = mask_Skull - segmented_GM_WM_CSF
		segmented_GM_WM_CSF_Skull = np.where(segmented_GM_WM_CSF_Skull<0, 4, segmented_GM_WM_CSF_Skull)#Consider the small overlap positions as Skull
		#Re-labelling tissues
		segmented_GM_WM_CSF_Skull += 5
		segmented_GM_WM_CSF_Skull = np.where(segmented_GM_WM_CSF_Skull==5, 0 , segmented_GM_WM_CSF_Skull)
		segmented_GM_WM_CSF_Skull = np.where(segmented_GM_WM_CSF_Skull==9, 4 , segmented_GM_WM_CSF_Skull)
		segmented_GM_WM_CSF_Skull = np.where(segmented_GM_WM_CSF_Skull==6, 3 , segmented_GM_WM_CSF_Skull)
		segmented_GM_WM_CSF_Skull = np.where(segmented_GM_WM_CSF_Skull==7, 2 , segmented_GM_WM_CSF_Skull)
		segmented_GM_WM_CSF_Skull = np.where(segmented_GM_WM_CSF_Skull==8, 1 , segmented_GM_WM_CSF_Skull)


		#Process Scalp
		#Find Scalp largest contour
		Scalp = Scalp.astype(np.uint8)
		Scalp_contours,hierarchy = cv2.findContours(Scalp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour_lengths = []
		for i in range(0,len(Scalp_contours)):
			contour_lengths.append(len(Scalp_contours[i]))
		max_index = contour_lengths.index(max(contour_lengths))
		#Generate the GM mask by filling all the pixels inside the mask
		img_contours = np.zeros((Scalp.shape[0],Scalp.shape[1]))
		contour_Scalp = cv2.drawContours(img_contours, [Scalp_contours[max_index]], 0, (5), thickness = 1)
		img_contours = np.zeros((Scalp.shape[0],Scalp.shape[1]))
		mask_Scalp = cv2.drawContours(img_contours, [Scalp_contours[max_index]], 0, (5), thickness = -1)
		segmented_GM_WM_CSF_Skull_Scalp = mask_Scalp - segmented_GM_WM_CSF_Skull
		segmented_GM_WM_CSF_Skull_Scalp = np.where(segmented_GM_WM_CSF_Skull_Scalp<0, 5, segmented_GM_WM_CSF_Skull_Scalp)#Consider the small overlap positions as Scalp
		#Re-labelling tissues
		segmented_GM_WM_CSF_Skull_Scalp += 5
		segmented_GM_WM_CSF_Skull_Scalp = np.where(segmented_GM_WM_CSF_Skull_Scalp==5, 0 , segmented_GM_WM_CSF_Skull_Scalp)
		segmented_GM_WM_CSF_Skull_Scalp = np.where(segmented_GM_WM_CSF_Skull_Scalp==10, 5 , segmented_GM_WM_CSF_Skull_Scalp)
		segmented_GM_WM_CSF_Skull_Scalp = np.where(segmented_GM_WM_CSF_Skull_Scalp==6, 4 , segmented_GM_WM_CSF_Skull_Scalp)
		segmented_GM_WM_CSF_Skull_Scalp = np.where(segmented_GM_WM_CSF_Skull_Scalp==7, 3 , segmented_GM_WM_CSF_Skull_Scalp)
		segmented_GM_WM_CSF_Skull_Scalp = np.where(segmented_GM_WM_CSF_Skull_Scalp==8, 2 , segmented_GM_WM_CSF_Skull_Scalp)
		segmented_GM_WM_CSF_Skull_Scalp = np.where(segmented_GM_WM_CSF_Skull_Scalp==9, 1 , segmented_GM_WM_CSF_Skull_Scalp)


		'''
		############################ Plot the results #############################
		def grid_plot(grid,ind,im,title):
			im = grid[ind-1].imshow(im)
			#grid[ind].title(title,fontsize=10)
			grid[ind-1].set_xticks([])
			grid[ind-1].set_yticks([])
			grid[ind-1].text(225,30,str(ind),fontweight='bold',color='white')
			grid[ind-1].text(10,245,title,fontweight='bold',color='black',fontsize=8,backgroundcolor='white')

		fig = plt.figure(figsize=(20,10))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
			nrows_ncols=(3,6),  # creates 2x2 grid of axes
			axes_pad=0,  # pad between axes in inch.
			)

		grid_plot(grid,1,image,'Synthetic Slice')
		grid_plot(grid,2,WM,'Segmented WM')
		grid_plot(grid,3,WM_dialation2,'Eroded/Dialated WM')
		grid_plot(grid,4,GM,'Segmented GM')
		grid_plot(grid,5,GM_dialation,'Dialated GM')
		grid_plot(grid,6,contour_GM,'Outer Contour-GM')
		grid_plot(grid,7,segmented_GM_WM,'WM and GM merge')
		grid_plot(grid,8,CSF,'Segmented CSF')
		grid_plot(grid,9,CSF_dialation,'Dialated CSF')
		grid_plot(grid,10,contour_CSF,'Outer Contour-CSF')
		grid_plot(grid,11,segmented_GM_WM_CSF,'WM/GM/CSF')
		grid_plot(grid,12,Skull,'Skull')
		grid_plot(grid,13,Skull_dialation,'Dialated Skull')
		grid_plot(grid,14,contour_Skull,'Outer Contour-Skull')
		grid_plot(grid,15,segmented_GM_WM_CSF_Skull,'WM/GM/CSF/Skull')
		grid_plot(grid,16,Scalp,'Scalp')
		grid_plot(grid,17,contour_Scalp,'Outer Contour-Scalp')
		grid_plot(grid,18,segmented_GM_WM_CSF_Skull_Scalp,'WM/GM/CSF/Skull/Scalp')
		'''
		segmented_GM_WM_CSF_Skull_Scalp = np.reshape(segmented_GM_WM_CSF_Skull_Scalp,[1]+list(segmented_GM_WM_CSF_Skull_Scalp.shape))
		#plt.show()
		#plt.savefig(save_path+'.png')
		np.savez(save_path+'.npz',segmented_GM_WM_CSF_Skull_Scalp)
		#plt.close()
def Main():
	parser = argparse.ArgumentParser(description='Synthetic Head Model Generation',
									usage='Generate artificial head phantoms using GAN network',
									epilog='Give proper arguments')
	parser.add_argument('-ep',"--epoch",metavar='',help="Define the epoch to process the data",default=None)
	parser.add_argument('-f',"--res_fold",metavar='',help="Save training and testing results in folder",default='Resultant_Folder')

	args = parser.parse_args()
	ep = int(args.epoch)
	folder = args.res_fold

	folder = folder + '/Analysis/Evaluating_Images'
	epochs = os.listdir(folder)
	epochs = [int(ep) for ep in epochs]
	epochs.sort()
	print(epochs)

	if ep:
		epochs = [epochs[-1]]

	print('Selected Epochs = ',epochs)
	epoch_count = 0
	file_count = 0
	for i in range(0,len(epochs)):
		ep = epochs[i]
		directory = folder + '/' + str(ep)
		generated_images = os.listdir(directory)
		for im in generated_images:
			print('Sl = ',epoch_count,' Epoch = ',ep,' Image = ',file_count)
			try:
				file_path = directory + '/' + im
				save_path = directory + '/' + im.split('.')[0] + '_processed'
				process_generated_image(file_path,save_path)
				file_count += 1
			except Exception as ex:
				print(ex)
				print('Sl = ',epoch_count,' Epoch = ',ep,' Image = ',file_count,'----- Unsuccessful')
				file_count += 1
				continue
		epoch_count += 1
Main()

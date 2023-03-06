import os
import sys
import config
global C
C = config.Config()

def Folder_creation(Resultant_Folder):
	#Check whether specific folders are present or not....if not create them
	Folder_List = [
				'Training_History',
				'Training_Weights',
				'Model_Figures',
				'Tensorboard_Visualization',
				'Inference_Result',
				'Analysis']
	if not os.path.exists(Resultant_Folder):
		os.makedirs(Resultant_Folder)
	for i in range(0,len(Folder_List)):
		if not os.path.exists(Resultant_Folder+'/'+Folder_List[i]):
			os.makedirs(Resultant_Folder+'/'+Folder_List[i])
		if Folder_List[i]=='Analysis':
			try:
				os.makedirs(Resultant_Folder+'/'+Folder_List[i]+'/Pickle_Files')
				os.makedirs(Resultant_Folder+'/'+Folder_List[i]+'/detailed_prob')
				os.makedirs(Resultant_Folder+'/'+Folder_List[i]+'/Maximum_Prob')
				os.makedirs(Resultant_Folder+'/'+Folder_List[i]+'/Confusion_Matrix')
			except:
				print('Folder already exists')

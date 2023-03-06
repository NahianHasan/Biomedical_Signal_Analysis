import config
from scipy.io import loadmat
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,LabelBinarizer
from tensorflow import keras
from tensorflow.keras.utils import np_utils
import data_balancing as DB
global C
C = config.Config()

def data_standardization(X):
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	return X

def data_encoding(Y):
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoder_Y = encoder.transform(Y)
	#convert integers to dummy variables(i.e: one hot encoding)
	Y = np_utils.to_categorical(encoder_Y)
	return Y

def Modified_Feature_Data_Read(filepath_train,feature_numbers):
	#Feature read...... If already features are extracted, read them from the Modified Directory.
	#Be careful about data acquisition. If a large amount of data are loaded at once, the RAM might exhaust.
	X_train = list()
	Y_train = list()

	class_folders = os.listdir(filepath_train)
	for class_name in class_folders:
		class_files = os.listdir(filepath+'/'+class_name)
		for i in range(0,len(class_files)):
			Data = np.load(class_files[i],allow_pickle=True)
			features = np.array(Data['M'])
			if len(feature_number > 1):
				specific_feature = features[feature_number,:]
			else:
				specific_feature = features[feature_number[0],:]
			X_train.append(specific_feature)
			Y_train.append(class_name)

	return X_train, Y_train

def Data_Read(filepath,features):

	X_train,Y_train = Modified_Feature_Data_Read(filepath,features)

	X_train,Y_train = np.array(X_train), np.array(Y_train)

	print(Counter(Y_train.flatten()))

	#Data Process
	#Train dataset Standardization
	X_train = data_standardization(X_train)
	#encode class_values as integers
	Y_train = data_encoding(Y_train)

	if C.data_balance:
		print('\nData is being balanced.....Wait until it is finished\n')
		X_train,Y_train = DB.Data_Balancing(X_train,Y_train)
		Y_train = np.array(Y_train)
		print(Counter(Y_train.flatten()))

	return X_train, Y_train

import config
import utils as UL
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle
from numpy import ones,zeros
from numpy.random import rand,randint
global C
C = config.Config()

def Train_Single_Network(folder, g_model, d_model, gan_model, dataset, latent_dim=C.latent_dimension, n_epochs=C.nb_epochs, n_batch=C.nb_batches, n_eval=C.nb_evals,res_iter=C.initial_iter,resume_status=C.resume_status):
	# train the generator and discriminator
	bat_per_epo = int(len(dataset) / n_batch)
	half_batch = int(n_batch / 2)
	# calculate the total iterations based on batch and epoch
	if resume_status:
		initial_step = int(res_iter)
	else:
		initial_step = 0
	n_steps = bat_per_epo * n_epochs
	T=open(folder+'/Training_History/Histogram_Data.csv','w')
	T.write('Iteration,d_loss_real,d_loss_fake,gan_loss,d_accuracy_real,d_accuracy_fake\n')
	# manually enumerate epochs
	for i in range(initial_step,n_steps):
		# get randomly selected ✬ real ✬ samples
		X,Y = UL.generate_real_samples(dataset, half_batch)
		# update discriminator model weights
		d_loss1, d_acc1 = d_model.train_on_batch(X,Y)
		# generate ✬ fake ✬ examples
		X,Y = UL.generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
		d_loss2, d_acc2 = d_model.train_on_batch(X,Y)

		# prepare points in latent space as input for the generator

		X = UL.generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		Y = ones((n_batch, 1))
		# update the generator via the discriminator ✬ s error
		g_loss,g_acc = gan_model.train_on_batch(X,Y)
		#tensorboard_logg
		tensorboard.set_model(gan_model)
		tensorboard.on_epoch_end(i,UL.named_logs(['gan_loss','gan_acc','dis_loss_real','dis_acc_real','dis_loss_fake','dis_acc-fake'],[g_loss,g_acc,d_loss1,d_acc1,d_loss2,d_acc2]))
		# summarize loss on this batch
		print('Ep:',round((n_epochs*i)/n_steps,2) ,' iter:',i+1,'/',n_steps,' d_l_real:', round(d_loss1,5),' d_l_fake:', round(d_loss2,5),' Gan Loss:', round(g_loss,5),' d_ac_r:', round(int(100*d_acc1),5),' d_ac_f:', round(int(100*d_acc2),5))
		# record history
		T.write(str(i)+','+str(d_loss1)+','+str(d_loss2)+','+str(g_loss)+','+str(d_acc1)+','+str(d_acc2)+'\n')
		# evaluate the model every n_eval epochs
		if (i+1) % n_eval == 0:
			UL.summarize_performance(folder, i, g_model, d_model, gan_model)
	T.close()
	UL.plot_history(folder)

# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid ' unique layer name ' issue
			layer.name = ' ensemble_ ' + str(i+1) + ' _ ' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(10, activation= ' relu ' )(merge)
	output = Dense(3, activation= ' softmax ' )(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
	plot_model(model, show_shapes=True, to_file= ' model_graph.png ' )
	# compile
	model = MC.Model_Compile(model)

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# encode output data
	inputy_enc = to_categorical(inputy)
	# fit model
	history = model.fit(X, inputy_enc, epochs=300, verbose=0)
	return history

def Train(model,X_train, Y_train,X_val,Y_val,nb_epoch,batch,callbacks,initial_epoch,folder):

	ensembling_path = {
		'Horizontal_Ensembling':folder+'/Ensemble_Models/Horizontal_Ensembling/',
		'Snapshot_Ensembling':folder+'/Ensemble_Models/Snapshot_Ensembling/',
		'Multiple_Model_Ensembling':folder+'/Ensemble_Models/Multiple_Model_Ensembling/',
		'Same_Model_Ensembling':folder+'/Ensemble_Models/Same_Model_Ensembling/'
	}

	if C.training_type == 'Single_Network':
		model = Train_Single_Network(model,X_train, Y_train,X_val,Y_val,nb_epoch,batch,callbacks,initial_epoch)
		filehandler_history = open(folder+'/Training_History/History.obj','wb')
		pickle.dump(model.history,filehandler_history)
		filehandler_history.close()

	elif C.training_type == 'Horizontal_Ensembling':
		for i in range(C.nb_epoch):
			model = Train_Single_Network(model,X_train, Y_train,X_val,Y_val,nb_epoch,batch,callbacks,initial_epoch)
			if i>=C.H_ensembling['nb_epoch_save_after']:
				filehandler_history = open(ensembling_path[C.training_type]+'model_' + str(i) + '.obj','wb')
				pickle.dump(model.history,filehandler_history)
				filehandler_history.close()
		return model

	elif C.training_type == 'Snapshot_Ensembling':
		return Train_Single_Network(model,X_train, Y_train,X_val,Y_val,nb_epoch,batch,callbacks,initial_epoch)

	elif C.training_type == 'Multiple_Model_Ensembling':
		all_models = []
		for i in range(0,len(C.M_model_ensembling['models'])):
			single_model = Train_Single_Network(model,X_train, Y_train,X_val,Y_val,nb_epoch,batch,callbacks,initial_epoch)
			filehandler_history = open(ensembling_path[C.training_type]+C.M_model_ensembling['models'][i] + '.obj','wb')
			pickle.dump(single_model.history,filehandler_history)
			filehandler_history.close()
			all_models.append(single_model)
		return all_models

	elif C.training_type == 'Same_Model_Ensembling':
		all_models = []
		for i in range(0,C.S_model_ensembling['number_of_models']):
			single_model = Train_Single_Network(model,X_train, Y_train,X_val,Y_val,nb_epoch,batch,callbacks,initial_epoch)
			filehandler_history = open(ensembling_path[C.training_type]+'model_'+ str(i) + '.obj','wb')
			pickle.dump(single_model.history,filehandler_history)
			filehandler_history.close()
			all_models.append(single_model)
		return all_models

	if C.ensemble_combination_type == 'stacked_generalization':
		models_directory = ensembling_path[C.training_type]
		members = TEN.load_all_models(models_directory)
		integrated_stacked_model = define_stacked_model(members)
		# fit stacked model on test dataset
		integrated_stacked_model_history = fit_stacked_model(stacked_model, train_X, train_Y)
		filehandler_history = open(ensembling_path[C.training_type]+'stacked_model.obj','wb')
		pickle.dump(integrated_stacked_model_history.history,filehandler_history)
		filehandler_history.close()

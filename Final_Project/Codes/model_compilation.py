import tensorflow.keras.optimizers as optim
import config
global C
C = config.Config()


def Model_Compile(model,optimizer,loss_type):
	Opt = {
		'sgd':optim.SGD(learning_rate=C.initial_lrate, momentum=0.9, decay=1e-6, nesterov=True),
		'adam':optim.Adam(learning_rate=C.initial_lrate, beta_1 = 0.5, beta_2 = 0.99, decay = 0.00001),
		'RMSprop':optim.RMSprop(learning_rate=C.initial_lrate, rho=0.9),
		'adadelta':optim.Adadelta(learning_rate=C.initial_lrate, rho=0.95, epsilon=1e-07),
		'adagrad':optim.Adagrad(learning_rate=C.initial_lrate,epsilon=1e-07),
		'adamax':optim.Adamax(learning_rate=C.initial_lrate, beta_1=0.5, beta_2=0.999, epsilon=1e-07),
		'nadam':optim.Nadam(learning_rate=C.initial_lrate, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
	}
	model.compile(loss = loss_type, optimizer = Opt[optimizer],metrics=C.metrics)

	return model

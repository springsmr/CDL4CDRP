"""Implementation of the discriminative RBM in Theano which is trained using 
gradient descent."""


# Author: Srikanth Cherla
# City University London (2014)
# Contact: abfb145@city.ac.uk


import numpy as np
import theano
import theano.tensor as T
import time

from evaluate import accuracy
from evaluate import negative_log_likelihood
from IO import generate_file_name
from optimize import sgd
from utils import make_batches

theano.config.exception_verbosity = 'high'


#############################
# Section: Model definition #
#############################
class DRBM(object):
    """Discriminative restricted Boltzmann machine class"""

    def __init__(self, n_input, n_class, hypers, init_params=None):
        """Constructs and compiles Theano functions for learning and
        prediction.

        Input
        -----
        n_input : integer
          Number of inputs to the model
        n_class : integer
          Number of outputs
        hypers : dictionary
          Model hyperparameters
        init_params : list
          Model initial parameters
        """
        self.model_type = str(hypers['model_type'])
        self.n_input = n_input
        self.n_class = n_class
        self.n_hidden = int(hypers['n_hidden'])
        self.L1_reg = float(hypers['weight_decay'])
        self.L2_reg = float(hypers['weight_decay'])
        self.activation = str(hypers['activation'])
        self.bin_size = int(hypers['bin_size'])
        self.loss = str(hypers['loss'])
        self.seed = float(hypers['seed'])
        self.uid = time.strftime('%Y-%m-%d-%H-%M-%S') + \
            generate_file_name('', hypers, '', '')

        # Build the model graph.
        (x, y, y_pred, p_y_given_x, cost, params, last_updates, energies, 
            log_p) = build_drbm(n_input, n_class, hypers, init_params)

        # Compute parameter gradients w.r.t cost.
        lr = T.scalar('learning_rate', dtype=theano.config.floatX)
        mom = T.scalar('momentum', dtype=theano.config.floatX)

        gradients = T.grad(cost, params)
        updates_train = []
        for (update, param, gradient) in zip(last_updates, params, gradients):
            updates_train.append((param, param + mom*update - lr*gradient))
            updates_train.append((update, mom*update - lr*gradient))

        # Functions for training, evaluating and saving the model.
        self.get_model_parameters = theano.function([], params)
        self.train_function = theano.function([x, y, lr, mom], cost,
                                              updates=updates_train,
                                              allow_input_downcast=True)
        self.predict_function = theano.function([x, ], y_pred,
                                                allow_input_downcast=True)
        self.predict_proba = theano.function([x, ], p_y_given_x,
                                             allow_input_downcast=True)
        self.get_logp = theano.function([x, ], log_p,
                                            allow_input_downcast=True)
        self.get_energies = theano.function([x, ], energies,
                                            allow_input_downcast=True)


def build_drbm(n_input, n_class, hypers, init_params):
    """Function to build the Theano graph for the DRBM.

    Input
    -----
    n_input : integer
      Dimensionality of input features to the model.
    n_class : integer
      Number of class-labels.
    hypers : dict
      Model hyperparameters.
    init_params : list
      A list of initial values for the model parameters.

    Output
    ------
    x: T.matrix
      Input matrix (with number of data points as first dimension).
    y: T.ivector
      Class labels corresponding to x.
    y_pred: T.argmax
      Predictions (class-labels) corresponding to p_y_given_x.
    p_y_given_x: T.nnet.softmax
      Posterior probability of y given x.
    cost: ???
      Cost function of the DRBM which is to be optimized.
    params: list
      A list containing the four parameters of the DRBM (see class definition).
    """
    n_hidden = int(hypers['n_hidden'])
    L1_reg = float(hypers['weight_decay'])
    L2_reg = float(hypers['weight_decay'])
    activation = str(hypers['activation'])
    bin_size = int(hypers['bin_size'])
    RNG = np.random.RandomState(hypers['seed'])

    # 1. Initialize inputs and targets
    x = T.matrix(name='x', dtype=theano.config.floatX)

    # 2. Initialize outputs
    y = T.ivector(name='y')

    # 3. Initialize model parameters
    if init_params is None:
        U_init = np.asarray((RNG.rand(n_class, n_hidden) * 2 - 1) /
                            np.sqrt(max(n_class, n_hidden)),
                            dtype=theano.config.floatX)
        W_init = np.asarray((RNG.rand(n_input, n_hidden) * 2 - 1) /
                            np.sqrt(max(n_input, n_hidden)),
                            dtype=theano.config.floatX)
        c_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        d_init = np.zeros((n_class,), dtype=theano.config.floatX)
    else:
        U_init = init_params[0]
        W_init = init_params[1]
        c_init = init_params[2]
        d_init = init_params[3]

    U = theano.shared(U_init, name='U')  # Class-hidden weights
    W = theano.shared(W_init, name='W')  # Input-hidden weights
    c = theano.shared(c_init, name='c')  # Hidden biases
    d = theano.shared(d_init, name='d')  # Class biases
    
    params = [U, W, c, d]

    # Most recent updates to use with momentum
    last_updates = [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                  dtype=theano.config.floatX)) 
                    for param in params]

    # Predict posterior probabilities and class-labels
    p_y_given_x, energies, log_p = drbm_fprop(x, params, n_class, activation,
                                              bin_size)
    y_pred = T.argmax(p_y_given_x, axis=1)

    # Loss functions
    Y_class = theano.shared(np.eye(n_class, dtype=theano.config.floatX),
                            name='Y_class')
    if hypers['loss'] == 'll': # Log-likelihood
        loss = -T.mean(T.sum(T.log(p_y_given_x) * Y_class[y], axis=1))
    elif hypers['loss'] == 'ce': # Cross-entropy
        loss = -T.mean(T.sum(T.log(p_y_given_x) * Y_class[y], axis=1) + \
                       T.sum(T.log(1-p_y_given_x) * (1-Y_class[y]), axis=1))
    elif hypers['loss'] == 'se': # Squared-error
        loss = T.mean(T.sum((Y_class[y] - p_y_given_x) ** 2, axis=1))

    # Regularization with L1 and L2 norms
    L1 = abs(W).sum() + abs(U).sum()
    L2 = (W**2).sum() + (U**2).sum()

    cost = loss + L1_reg*L1 + L2_reg*L2

    return x, y, y_pred, p_y_given_x, cost, params, last_updates, energies, \
        log_p 


def drbm_fprop(x, params, n_class, activation, bin_size):
    """Posterior probability of classes given inputs and model parameters.

    Input
    -----
    x: T.matrix (of type theano.config.floatX)
      Input data matrix.
    params: list
      A list containing the four parameters of the DRBM (see class definition).
    n_class: integer
      Number of classes.

    Output
    ------
    p_y_given_x: T.nnet.softmax
      Posterior class probabilities of the targets given the inputs.
    """
    # Initialize DRBM parameters and binary class-labels.
    U = params[0]
    W = params[1]
    c = params[2]
    d = params[3]
    Y_class = theano.shared(np.eye(n_class, dtype=theano.config.floatX),
                            name='Y_class')

    # Compute hidden state activations and energies.
    s_hid = T.dot(x, W) + c
    energies, _ = theano.scan(lambda y_class, U, s_hid:
                              s_hid + T.dot(y_class, U),
                              sequences=[Y_class],
                              non_sequences=[U, s_hid])

    max_energy = T.max(energies)

    # Compute log-posteriors and then posteriors.
    if activation == 'basic':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + T.sum(T.log(1+T.exp(e_i)), axis=1),
            sequences=[d, energies], non_sequences=[])
    elif activation == 'bipolar':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + T.sum(T.log(T.exp(-e_i)+T.exp(e_i)), axis=1),
            sequences=[d, energies], non_sequences=[])
    elif activation == 'binomial':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + \
            T.sum(T.log((1-T.exp(bin_size*e_i))/(1-T.exp(e_i))), axis=1),
            sequences=[d, energies], non_sequences=[])
    else:
        raise NotImplementedError

    p_y_given_x = T.nnet.softmax(log_p.T)

    return p_y_given_x, energies, log_p 


#####################
# Section: Training #
#####################
def train_model(dataset, mod_hypers, opt_hypers, dat_hypers,
                model_params=None):
    """Stochastic gradient descent optimization of a Discriminative RBM.

    Input
    -----
    dataset : tuple(tuple(np.ndarray))
      Training and validation sets
    mod_hypers : dictionary
      Model hyperparameters
    opt_hypers : dictionary
      Optimization hyperparameters
    model_params : list(np.ndarray)
      A list of model parameter values (optional)

    Output
    ------
    model_params : list(np.ndarray)
      List of all the model parameters (see class definition)
    valid_nll : float
      Validation set negative log-likelihood
    """
    # Read training, validation and test sets
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    if opt_hypers['eval_test'] == True:
        X_test, y_test = dataset[2]

    # We need these as well, and note that n_visible = n_input + n_class
    n_input = dat_hypers['n_input']
    n_class = dat_hypers['n_class']

    # Train the DRBM
    model = DRBM(n_input=n_input, n_class=n_class, hypers=mod_hypers,
                 init_params=model_params) 

    # The actual learning step
    if opt_hypers['opt_type'] == 'batch-gd': # Batch learning
        X_train, y_train = make_batches(X_train, y_train,
                                        opt_hypers['batch_size']) 
        X_valid, y_valid = make_batches(X_valid, y_valid,
                                        X_valid.shape[0])
        if opt_hypers['eval_test'] == True:
            X_test, y_test = make_batches(X_test, y_test,
                                            X_test.shape[0])
            dataset = ((X_train, y_train), (X_valid, y_valid), 
                       (X_test, y_test))
        else:
            dataset = ((X_train, y_train), (X_valid, y_valid))

        optimizer = sgd(opt_hypers)
        params, valid_score = optimizer.optimize(model, dataset)
    else:
        raise NotImplementedError

    return params, valid_score


#######################
# Section: Evaluation #
#######################
def test_model(model, data, eva_hypers):
    """Evaluate an already trained RNN on given test data.

    Input
    -----
    model : tuple(list(np.ndarray), dict)
      A tuple containing the model parameters, and a dictionary with its
      hyperparameters.
    dataset : tuple(tuple(np.ndarray), dict)
      A tuple containing test inputs and their corresponding labels, and
      dictionary with information about the data.
    eva_hypers: dict
      A dictionary which specifies how exactly to evaluate the model.
    """
    test_data, dat_hypers = data
    n_input = dat_hypers['n_input']
    n_class = dat_hypers['n_class']
    X_test = test_data[0]
    y_test = test_data[1]
    model_params, mod_hypers = model

    model = DRBM(n_input=n_input, n_class=n_class, hypers=mod_hypers,
                 init_params=model_params)

    # Cross entropy
    y_prob = model.predict_proba(X_test)
    test_nll = negative_log_likelihood(y_prob, y_test)

    # Accuracy
    y_pred = model.predict_function(X_test)
    test_acc = accuracy(y_pred, y_test)

    print "Test negative log-likelihood (offline): %.3f\n" % (test_nll)
    print "Test accuracy (offline): %.3f\n" % (test_acc)

    return y_prob, test_nll, y_pred, test_acc


if __name__ == '__main__':
    print "Did not implement a main function. Use with train_models.py."

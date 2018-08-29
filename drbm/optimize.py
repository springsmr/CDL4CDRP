"""Optimizers for gradient-based learning of model parameters.
"""

import cPickle
import numpy as np
import os
import sys
import theano.tensor as T

from evaluate import accuracy

RNG = np.random.RandomState(860331)


class sgd(object):
    """Batch gradient descent optimizer class definition.
    
    Optimization hyperparameters in use at the moment:
      learning_rate : float
        Step-size for model parameter updates (suggested value: 0.01).
      threshold : float
        Decay value for adaptive learning rate (suggested value: 1.0).
      max_epoch: int
        Maximum number of epochs (suggested value: 100).
      initial_momentum : float
        Nesterov momentum to speed up learning (suggested value: 0.5).
      final_momentum : float
        Nesterov momentum to speed up learning (suggested value: 0.9).
      momentum_switchover : int 
        Iteration in which to switch from initial to final momentum
        (suggested value: 5)
    """
    def __init__(self, hypers):
        """Constructs an sgd_optimizer with the given hyperparameters.

        Input
        -----
        hypers : dictionary
          Optimization hyperparameters. 
        """
        self.opt_type=str(hypers['opt_type'])
        self.learning_rate=float(hypers['learning_rate'])
        self.schedule=str(hypers['schedule'])
        self.threshold=int(hypers['threshold'])
        self.patience=int(hypers['patience'])
        self.max_epoch=int(hypers['max_epoch'])
        self.validation_frequency = int(hypers['validation_frequency'])
        self.initial_momentum=float(hypers['initial_momentum'])
        self.final_momentum=float(hypers['final_momentum'])
        self.momentum_switchover=int(hypers['momentum_switchover'])
        self.eval_test=bool(hypers['eval_test'])


    def optimize(self, model, dataset):
        """Learn a model using batch gradient descent.

        Input
        -----
        model : Python class 
          Definition of model.
        dataset : tuple(tuple(np.ndarray))
          Each tuple contains inputs and targets for training, validation and
          test data.

        Output
        ------
        best_model_params: list(np.ndarray)
          List of all the model parameters
        best_valid_acc: float
          Best validation set accuracy.
        """
        # Load training and validation data, and check for sparsity
        X_train, y_train = dataset[0]
        X_valid, y_valid = dataset[1]
        if self.eval_test == True:
            X_test, y_test = dataset[2]

        # Generate file name to save intermediate training models
        os.mkdir('.' + model.uid)
        temp_file_name = os.path.join('.' + model.uid, 'best_model.pkl')

        n_valid = len(X_valid)
        if self.eval_test == True:
            n_test = len(X_test)

        # Initialize learning rate and schedule
        learning_rate = self.learning_rate
        threshold = self.threshold
        if self.schedule == 'constant':
            rate_update = lambda coeff: self.learning_rate
        elif self.schedule == 'linear':
            rate_update = lambda coeff: self.learning_rate / (1+coeff)
        elif self.schedule == 'exponential':
            rate_update = lambda coeff: self.learning_rate / \
            10**(coeff/self.threshold)
        elif self.schedule == 'power':
            rate_update = lambda coeff: self.learning_rate / \
                (1 + coeff/self.threshold)
        elif self.schedule == 'inv-log':
            rate_update = lambda coeff: self.learning_rate / \
                (1+np.log(coeff+1)) 

        max_epoch = self.max_epoch
        validation_frequency = self.validation_frequency
        best_valid_acc = -1.0
        if self.eval_test == True:
            best_test_acc = -1.0
        cPickle.dump(model, open(temp_file_name, 'wb'))
        best_params = model.get_model_parameters()
        
        # Early stopping parameters and other checks
        patience = self.patience
        pissed_off = 0

        for epoch in xrange(max_epoch):

            # Set effective momentum for the current epoch.
            effective_momentum = self.final_momentum \
                if epoch > self.momentum_switchover \
                else self.initial_momentum
            
            # Check if it's time to stop learning.
            if threshold == 0:
                if pissed_off == patience: # Exit and return best model
                    model = cPickle.load(open(temp_file_name, 'rb'))
                    best_params = model.get_model_parameters()
                    
                    print('Learning terminated after %d epochs.\n' % (epoch+1))
                    break
                else: # Reload previous best model and continue
                    pissed_off+=1
                    learning_rate = rate_update(pissed_off)
                    threshold = self.threshold
                    model = cPickle.load(open(temp_file_name, 'rb'))

                    print('Re-initialising to previous best model with '
                          'validation prediction accuracy %.3f.\n'
                          '\tCurrent pissed off level: %d/%d.\n'
                          '\tCurrent learning rate: %.4f.\n'
                          % (best_valid_acc, pissed_off, patience, 
                             learning_rate))

            RNG.seed(0xbeef); RNG.shuffle(X_train)
            RNG.seed(0xbeef); RNG.shuffle(y_train)
            costs = []
            for X, y in zip(X_train, y_train):
                costs.append(model.train_function(X, y, learning_rate,
                                                  effective_momentum))
            mean_train_loss = np.mean(costs)
            
            print('Epoch %i/%i, train loss: %.3f bits' %
                  (epoch+1, max_epoch, mean_train_loss))

            if (epoch + 1) % validation_frequency == 0:
                # Compute validation negative log-likelihood
                valid_pred = []
                if self.eval_test == True:
                    test_pred = []

                for i in xrange(n_valid):
                    valid_pred.append(  # Don't pass labels
                        model.predict_function(X_valid[i]))
                this_valid_acc = accuracy( 
                    np.concatenate(tuple(valid_pred), axis=0),
                    np.concatenate(tuple(y_valid), axis=0))

                if self.eval_test == True:
                    for i in xrange(n_test):
                        test_pred.append(  # Don't pass labels
                            model.predict_function(X_test[i]))
                    this_test_acc = accuracy( 
                        np.concatenate(tuple(test_pred), axis=0),
                        np.concatenate(tuple(y_test), axis=0))

                print("\tValidation accuracy: %.3f (previous best: %.3f)" %
                      (this_valid_acc, best_valid_acc))
                if self.eval_test == True:
                    print("\tTest accuracy: %.3f (previous best: %.3f)" %
                          (this_test_acc, best_test_acc))
                print "\tCurrent learning rate: %.4f" % (learning_rate)

                if this_valid_acc > best_valid_acc:
                    best_valid_acc = this_valid_acc
                    best_params = model.get_model_parameters()
                    cPickle.dump(model, open(temp_file_name, 'wb'))
                    threshold = self.threshold
                else:
                    threshold-=1

        # Clean temporary files and folder
        os.remove(temp_file_name)
        os.rmdir('.' + model.uid)

        return best_params, best_valid_acc

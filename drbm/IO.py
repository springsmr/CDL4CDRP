import gzip
import cPickle
import sys
from utils import dict_combine

from os import path
from os import system


def generate_file_name(root, hyper_dict, prefix='model', ext='.pkl.gz'):
    """Generate file name from model hyperparameters.

    Input
    -----
    root : str
      Folder which contains the file.
    hyper_dict : OrderedDict
      Dictionary containing hyperparameters that go into the file name string.
    prefix : str
      File name prefix
    ext : str
      File extension

    Output
    ------
    file_name : str
      File name generated from root and hyper_dict.
    """
    file_name = path.join(root, prefix)
    for (h, v) in hyper_dict.items():
        file_name += ('-'+str(v))
    file_name += ext

    return file_name


def update_file_name(file_name, hyper_dict):
    """Update file name by adding new hyperparameters to name string.

    Input
    -----
    file_name : string
      Original file name.
    hyper_dict : OrderedDict
      Dictionary containing hyperparameters that are added to file name string.

    Output
    ------
    upd_file_name : string
      Updated file name.
    """
    fn_ext=''
    fn_prefix = file_name
    while True:
        fn_prefix, fn_sub_ext = path.splitext(fn_prefix)
        fn_ext = fn_sub_ext+fn_ext
        if fn_ext == '.pkl.gz' or fn_ext == '.pkl':
            break
    upd_file_name = fn_prefix + '--eval'
    for (h, v) in hyper_dict.items():
        upd_file_name += ('-'+str(v))
    upd_file_name += fn_ext

    return upd_file_name


def save_model(model, root):
    """Save a model into a folder.

    Input
    -----
    model : class(object)
      Instance of the class defining the Theano model.
    root  : string
      Path of the folder in which to save it.

    Output
    ------
      A cPickle file containing the model is written to the specified folder.
    """
    hyper_dict = dict_combine(model['mod_hypers'], model['opt_hypers'])
    file_name = generate_file_name(root, hyper_dict)
    dir_path = path.split(file_name)[0]

    if not path.exists(dir_path):
        system('mkdir -p ' + dir_path)

    cPickle.dump(model, gzip.open(file_name, 'wb'))
    print "Successfully saved model to the file:\n\t%s\n" % (file_name)


def load_model(file_name):
    """Load a model.
    
    Input
    -----
    file_name : string
      Name of cPickle file containing model information.

    Output
    ------
    _ : Python object
      An instance of the model contained in the cPickle file.
    """
    return cPickle.load(gzip.open(file_name, 'rb'))


def update_model(model, file_name):
    """Update a learned model with test scores.

    Input
    -----
    model : dict
      A dictionary with the learned model parameters, model and optimization
      hyperparameters.
    file_name : string
      Name of the file to save the model to.

    Output
    ------
      Updates the model in the specified file and saves it to a new file with
      the evaluation hyperparameters appended to the existing name.
    """
    root = path.split(file_name)[0]
    mod_opt_hyper = dict_combine(model['mod_hypers'], model['opt_hypers'])
    assert generate_file_name(root, mod_opt_hyper) == file_name, \
        "Generated file name does not match the input file name."

    new_filename = update_file_name(file_name, model['eva_hypers'])
    cPickle.dump(model, gzip.open(new_filename, 'wb'))
    
    print "Successfully updated the model in %s\n" % (file_name)


def model_exists(mod_hypers, opt_hypers, root):
    """Check if a specific model has already been learned and saved.
    """
    mod_opt_hyper = dict_combine(mod_hypers, opt_hypers)
    file_name = generate_file_name(root, mod_opt_hyper)
    return path.exists(file_name)

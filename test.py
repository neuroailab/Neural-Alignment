'''
Runs validations on neural-alignment models
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl.logging as _logging

import os
import tensorflow as tf
import numpy as np

from tfutils import base

import flags
from params import Params

FLAGS = flags.FLAGS

# Main test_form_params setup to re-validate a checkpointed model
def validate(test_params):
    print("Validating only")
    print("All params: ")
    print(test_params)
    base.test_from_params(**test_params)

# Main test_form_params setup to re-validate a checkpointed model on TPU
def validate_tpu(test_params):
    print("Validating only")
    print("All params: ")
    print(test_params)
    base.train_from_params(**test_params)

def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    params = Params()
    if FLAGS.load_params_file is not None:
        assert(FLAGS.load_params_file[-4:] == '.pkl')
        print("Loading params from file: {}".format(FLAGS.load_params_file))
        print("Ignoring all config flags")
        params.load(FLAGS.load_params_file, FLAGS)
    else:
        params.customize(flags=FLAGS)
    test_params = params.get_params_copy()
    if argv[1] != 'validate_tpu':
        test_params.pop('train_params')
        test_params.pop('loss_params')
        test_params.pop('optimizer_params')
        test_params.pop('learning_rate_params')
    test_params['save_params'].pop('save_valid_freq')
    test_params['save_params'].pop('save_filters_freq')
    test_params['save_params'].pop('cache_filters_freq')
    # exclude global step during validation, will throw error since no
    # global_step variable exists during validation mode
    test_params['validation_params']['topn_val']['targets']['include_global_step'] = False

    test_params['load_params'] = test_params['save_params']
    test_params['load_params']['do_restore'] = True
    if FLAGS.load_step is None:
        test_params['load_params']['query'] = None # load from the latest step
    else:
        test_params['load_params']['query'] = {'step': FLAGS.load_step}
    if FLAGS.load_checkpoint is not None:
        test_params['load_params'].pop('query', None)
        test_params['load_params'].update({'from_ckpt': FLAGS.load_checkpoint})

    orig_exp_id = test_params['save_params']['exp_id']
    assert FLAGS.val_exp_id is not None
    test_params['save_params'] = {'exp_id': FLAGS.val_exp_id, 'save_to_gfs': []}
    if FLAGS.save_to_gfs is not None:
        test_params['save_params']['save_to_gfs'] = FLAGS.save_to_gfs.split(',')
    if FLAGS.load_port is not None: # load from a different db but save to a new one
        test_params['load_params']['port'] = FLAGS.load_port
        test_params['save_params']['port'] = FLAGS.port

    if argv[1] == 'validate':
        validate(test_params)
    if argv[1] == 'validate_tpu':
        validate_tpu(test_params)
    else:
        print("No known validation specified. Rerun with either 'validate' or 'validate_tpu'")

if __name__ == '__main__':
    tf.app.run(main)


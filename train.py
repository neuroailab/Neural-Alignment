'''
Trains neural-alignment models

All options should be configured via flags
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl.logging as _logging

import os
import tensorflow as tf

from tfutils import base

import flags
from params import Params

FLAGS = flags.FLAGS

#### Actual training
def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    del argv  # Unused
    params = Params()
    if FLAGS.load_params_file is not None:
        assert(FLAGS.load_params_file[-4:] == '.pkl')
        print("Loading params from file: {}".format(FLAGS.load_params_file))
        print("Ignoring all config flags")
        params.load(FLAGS.load_params_file, FLAGS)
    else:
        print("Parsing params from flags...")
        params.customize(flags=FLAGS)
    params_copy = params.get_params_copy()
    print("All params: ")
    print(params_copy)
    if FLAGS.save_params_file is not None:
        assert(FLAGS.save_params_file[-4:] == '.pkl')
        params.save(FLAGS.save_params_file)
    base.train_from_params(**params_copy)

if __name__ == '__main__':
    tf.app.run(main)

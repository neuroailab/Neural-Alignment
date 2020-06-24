'''
Builds a dictionary structured as tfutils expects it
'''

import copy
import tensorflow as tf
import dill

from tfutils import optimizer
# Data providers
from tfutils.imagenet_data import ImageNet
from tfutils.tests import mnist_data
from tfutils.db_interface import TFUTILS_HOME

import learning_rate as lr
import rate_scheduler
from Alignments import alignment
from Metrics import functions as metrics
from Metrics import losses
from Models import basic, resnet_model_google
from custom_optimizers import SWATSOptimizer, RAdamOptimizer
from custom_optimizers import build_noisy_optimizer

class Params:
    def __init__(self):
        self.params = {}
        self.alignment_kwargs = {}
        self.rate_scheduler_kwargs = {}
        self.lr_scheduler_kwargs = {}
        self.alignment_lr_scheduler_kwargs = {}

    def _build_default_params(self, flags):
        FLAGS = flags
        # Dataset constants
        if FLAGS.dataset == 'mnist':
            self._LABEL_CLASSES = 10
            self._NUM_CHANNELS = 1
            self._NUM_TRAIN_IMAGES = 60000
            self._NUM_EVAL_IMAGES = 10000
            self.train_data_params = {
                'func': mnist_data.build_data,
                'batch_size': FLAGS.train_batch_size,
                'group': 'train',
                'directory': TFUTILS_HOME}
            self.val_data_params =  {
                'func': mnist_data.build_data,
                'batch_size': FLAGS.eval_batch_size,
                'group': 'test',
                'directory': TFUTILS_HOME}
        elif FLAGS.dataset == 'imagenet':
            self._LABEL_CLASSES = 1000
            self._NUM_CHANNELS = 3
            self._NUM_TRAIN_IMAGES = 1281167
            self._NUM_EVAL_IMAGES = 49920

            if FLAGS.use_resnet_v2:
                self._data_prep_type = 'inception'
            else:
                self._data_prep_type = 'resnet'

            # This is the GPU version of the data providers
            assert(FLAGS.data_dir is not None)
            self.train_data_params = {
                    'func': ImageNet(image_dir=FLAGS.data_dir,
                                     prep_type=self._data_prep_type,
                                     resize=FLAGS.image_size).dataset_func,
                    'is_train': True,
                    'batch_size': FLAGS.train_batch_size}
            self.val_data_params = {
                    'func': ImageNet(image_dir=FLAGS.data_dir,
                                     prep_type=self._data_prep_type,
                                     resize=FLAGS.image_size).dataset_func,
                    'is_train': False,
                    'batch_size': FLAGS.eval_batch_size,
                    'q_cap': FLAGS.eval_batch_size,
                    'file_pattern': 'validation-*'}


        self.NUM_BATCHES_PER_EPOCH = self._NUM_TRAIN_IMAGES / FLAGS.train_batch_size
        self._MOMENTUM = 0.9
        self.rate_scheduler_kwargs = {'loss_rate': FLAGS.loss_rate,
                                      'alignment_rate': FLAGS.alignment_rate,
                                      'delay_epochs': FLAGS.delay_epochs,
                                      'alternate_step_freq': FLAGS.alternate_step_freq,
                                      'constant_rate': FLAGS.constant_rate,
                                      'num_batches_per_epoch': self.NUM_BATCHES_PER_EPOCH}

        self.lr_scheduler_kwargs = {'learning_rate': FLAGS.learning_rate,
                                    'num_batches_per_epoch': self.NUM_BATCHES_PER_EPOCH,
                                    'train_batch_size': FLAGS.train_batch_size,
                                    'base_batch_size': FLAGS.base_batch_size,
                                    'rescale_lr': FLAGS.rescale_lr,
                                    'constant_lr': FLAGS.constant_lr,
                                    'warmup_epochs': FLAGS.warmup_epochs,
                                    'alternate_step_freq': FLAGS.alternate_step_freq,
                                    'delay_epochs': FLAGS.delay_epochs,
                                    'delay_epochs_offset': None}

        if FLAGS.load_step is None:
            load_query = None # loads most recent step
        else:
            load_query = {'step': FLAGS.load_step}

        self.params = {
            'save_params': {
                'host': 'localhost',
                'port': FLAGS.port,
                'dbname': FLAGS.dbname,
                'collname': None,
                'exp_id': None,
                'do_save': True,
                'save_valid_freq': (int)(FLAGS.epochs_per_checkpoint*self.NUM_BATCHES_PER_EPOCH),
                'save_filters_freq': (int)(FLAGS.epochs_per_checkpoint*self.NUM_BATCHES_PER_EPOCH),
                'cache_filters_freq': (int)(FLAGS.epochs_per_checkpoint*self.NUM_BATCHES_PER_EPOCH),
                'cache_dir': None,  # model directory for google cloud bucket
            },

            'load_params': {
                'do_restore': FLAGS.do_restore,
                'query': load_query
            },

            'model_params': {
                'func': None,
                'model_prefix': 'model_0',
                'num_classes': self._LABEL_CLASSES, # dataset dependent
                'layers_list': [],
                'num_gpus': len(FLAGS.gpu.split(',')),
                'devices': ['/gpu:%i' % idx for idx in range(len(FLAGS.gpu.split(',')))],
                # The following params only needed if tpu, passed as kwargs to model_fn
                # Will only train on TPU is tpu_name is not None
                'tpu_name': FLAGS.tpu_name,
                'gcp_project': FLAGS.gcp_project,
                'tpu_zone': FLAGS.tpu_zone,
                'num_shards': FLAGS.num_shards,
                'iterations_per_loop': FLAGS.iterations_per_loop,
            },

            'train_params': {
                'targets': {'func': losses.loss_metric,
                            'target': 'labels',
                            'rate': rate_scheduler.build_schedule(**self.rate_scheduler_kwargs)},
                'data_params': self.train_data_params,
                'num_steps': (int)(FLAGS.train_epochs*self.NUM_BATCHES_PER_EPOCH),  # number of steps to train
                'thres_loss': float('Inf'),
                'validate_first': FLAGS.validate_first, # You may want to turn this off at debugging
                'include_global_step': False,
            },

            'loss_params': {
                'targets': 'labels',
                'agg_func': losses.mean_loss_with_reg,
                'agg_func_kwargs': {'rate': rate_scheduler.build_schedule(**self.rate_scheduler_kwargs)},
                'loss_per_case_func': losses.category_loss,
            },

            'learning_rate_params': {
                'func': lr.build_lr_schedule(**self.lr_scheduler_kwargs)
            },

            'optimizer_params': {
                'optimizer': optimizer.ClipOptimizer,
                'optimizer_class': tf.train.GradientDescentOptimizer,
                'clip': FLAGS.grad_clip,
                'clipping_value': FLAGS.grad_clipping_value,
                'clipping_method': FLAGS.grad_clipping_method
            },

            'validation_params': {
                'topn_val': {
                    'data_params': self.val_data_params,
                    'targets': {
                        'func': metrics.metric_fn,
                        'num_classes': self._LABEL_CLASSES
                    },
                    'num_steps': self._NUM_EVAL_IMAGES // FLAGS.eval_batch_size,
                    'agg_func': metrics.concat_agg_func,
                    'online_agg_func': metrics.online_agg_append,
                }
            },
            'skip_check': FLAGS.skip_check,
        }

        if FLAGS.minibatch_size is not None:
            self.params['train_params']['minibatch_size'] = FLAGS.minibatch_size

        if FLAGS.load_db:
            self.params['load_params'] = {'host': 'localhost',
                                          'port': FLAGS.load_port,
                                          'dbname': FLAGS.load_dbname,
                                          'collname': FLAGS.load_collname,
                                          'exp_id': FLAGS.load_exp_id,
                                          'do_restore': True,
                                          'query': load_query}

    def _set_optimizer(self, optimizer_class):
        if optimizer_class == 'momentum':
            print("Using Momentum Optimizer")
            optimizer_params = {
                'optimizer_class': tf.train.MomentumOptimizer,
                'optimizer_kwargs': {'momentum': self._MOMENTUM, 'use_nesterov': True}
            }
        elif optimizer_class == 'adagrad':
            print("Using Adagrad Optimizer")
            optimizer_params = {
                'optimizer_class': tf.train.AdagradOptimizer
            }
        elif optimizer_class == 'rmsprop':
            print("Using RMSProp Optimizer")
            # Note: default momentum value for RMSProp is 0.0
            optimizer_params = {
                'optimizer_class': tf.train.RMSPropOptimizer
            }
        elif optimizer_class == 'adam':
            print("Using ADAM Optimizer")
            optimizer_params = {
                'optimizer_class': tf.train.AdamOptimizer
            }
        elif optimizer_class == 'swats':
            print("Using SWATS Optimizer")
            optimizer_params = {
                'optimizer_class': SWATSOptimizer,
                'optimizer_kwargs': {'rectified_adam': False,
                                     'include_global_step': True}
            }
        elif optimizer_class == 'swrats':
            print("Using SWATS Optimizer with RAdam")
            optimizer_params = {
                'optimizer_class': SWATSOptimizer,
                'optimizer_kwargs': {'rectified_adam': True,
                                     'include_global_step': True}
            }
        elif optimizer_class == 'radam':
            print("Using RADAM Optimizer")
            optimizer_params = {
                'optimizer_class': RAdamOptimizer,
                'optimizer_kwargs': {'include_global_step': True}
            }
        else:
            print("Using default Gradient Descent Optimizer")
            optimizer_params = {
                'optimizer_class': tf.train.GradientDescentOptimizer
            }

        return optimizer_params

    def customize(self, flags):
        FLAGS = flags
        self._build_default_params(FLAGS)

        # Choosing the model
        if FLAGS.model == 'fc':
            self.params['model_params']['func'] = basic.fc
            self.params['model_params']['activation'] = FLAGS.activation
            self.params['save_params']['collname'] = 'fc'
            if FLAGS.layers_list is not None:
                layers_list = [int(l) for l in FLAGS.layers_list.split(",")]
                assert type(layers_list) == list
                for l in layers_list:
                    assert type(l) == int
                self.params['model_params']['layers_list'] = layers_list
                self.params['save_params']['collname'] = 'fc_' + str(FLAGS.layers_list.replace(',','-'))
        elif 'resnet' in FLAGS.model:
            self.params['save_params']['collname'] = FLAGS.model
            if FLAGS.use_resnet_v2:
                self.params['save_params']['collname'] += 'v2'
            self.params['model_params']['func'] = resnet_model_google.google_resnet_func
            self.params['model_params']['tf_layers'] = FLAGS.tf_layers
            self.params['model_params']['resnet_size'] = (int)(FLAGS.model.split('resnet')[-1])
            self.params['model_params']['use_v2'] = FLAGS.use_resnet_v2
            self.params['model_params']['bn_trainable'] = FLAGS.bn_trainable
            print("regularize_weights_via_model", FLAGS.regularize_weights_via_model)
            print("regularize_weights_via_model (type)", type(FLAGS.regularize_weights_via_model))
            self.params['model_params']['regularize_weights'] = FLAGS.regularize_weights_via_model

        self.params['save_params']['exp_id'] = str(FLAGS.alignment)
        self.params['save_params']['exp_id'] += FLAGS.exp_id_suffix

        # Common alignment kwargs used by the Alignment parent class
        self.alignment_kwargs = {
            'update_forward': FLAGS.update_forward,
            'input_distribution': FLAGS.input_distribution,
            'input_stddev': FLAGS.input_stddev,
            'use_bias_forward': FLAGS.use_bias_forward,
            'use_bias_backward': FLAGS.use_bias_backward,
            'activation_fn_override': FLAGS.activation_fn_override,
            'activation_forward': FLAGS.activation_forward,
            'activation_backward': FLAGS.activation_backward,
            'batch_center_backward_input': FLAGS.batch_center_backward_input,
            'center_input': FLAGS.center_input,
            'normalize_input': FLAGS.normalize_input,
            'batch_center_forward_output': FLAGS.batch_center_forward_output,
            'center_forward_output': FLAGS.center_forward_output,
            'normalize_forward_output': FLAGS.normalize_forward_output,
            'center_backward_output': FLAGS.center_backward_output,
            'normalize_backward_output': FLAGS.normalize_backward_output}

        # Alignment coefficient kwargs used to build rate schedulers
        self.alignment_coefficient_kwargs = {
            'alpha': {'start': FLAGS.alpha_start,
                      'stop': FLAGS.alpha_stop,
                      'cycle': FLAGS.alpha_cycle,
                      'schedule_rate': FLAGS.alpha_schedule_rate,
                      'schedule_type': FLAGS.alpha_schedule_type,
                      'num_batches_per_epoch': self.NUM_BATCHES_PER_EPOCH,
                      'train_epochs': FLAGS.train_epochs},
            'beta': {'start': FLAGS.beta_start,
                     'stop': FLAGS.beta_stop,
                     'cycle': FLAGS.beta_cycle,
                     'schedule_rate': FLAGS.beta_schedule_rate,
                     'schedule_type': FLAGS.beta_schedule_type,
                     'num_batches_per_epoch': self.NUM_BATCHES_PER_EPOCH,
                     'train_epochs': FLAGS.train_epochs},
            'gamma': {'start': FLAGS.gamma_start,
                      'stop': FLAGS.gamma_stop,
                      'cycle': FLAGS.gamma_cycle,
                      'schedule_rate': FLAGS.gamma_schedule_rate,
                      'schedule_type': FLAGS.gamma_schedule_type,
                      'num_batches_per_epoch': self.NUM_BATCHES_PER_EPOCH,
                      'train_epochs': FLAGS.train_epochs}
            }
        # Set the alignment
        if FLAGS.alignment == 'feedback':
            print("Using Feedback Alignment")
            self.params['model_params']['alignment'] = alignment.Feedback()
        elif FLAGS.alignment == 'symmetric':
            print("Using Symmetric Alignment")
            # alpha scheduler
            self.alignment_coefficient_kwargs['alpha']['value'] = FLAGS.alpha if FLAGS.alpha is not None else 1.0e-3
            self.alignment_kwargs['alpha'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['alpha'])
            # beta scheduler
            self.alignment_coefficient_kwargs['beta']['value'] = FLAGS.beta if FLAGS.beta is not None else 2.0e-3
            self.alignment_kwargs['beta'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['beta'])
            # build alignment class
            self.params['model_params']['alignment'] = alignment.Symmetric(**self.alignment_kwargs)
        elif FLAGS.alignment == 'activation':
            print("Using Activation Alignment")
            # alpha scheduler
            self.alignment_coefficient_kwargs['alpha']['value'] = FLAGS.alpha if FLAGS.alpha is not None else 1.0e-3
            self.alignment_kwargs['alpha'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['alpha'])
            # beta scheduler
            self.alignment_coefficient_kwargs['beta']['value'] = FLAGS.beta if FLAGS.beta is not None else 2.0e-3
            self.alignment_kwargs['beta'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['beta'])
            # build alignment class
            self.params['model_params']['alignment'] = alignment.Activation(**self.alignment_kwargs)
        elif FLAGS.alignment == 'mirror':
            print("Using Weight Mirror")
            # alpha scheduler
            self.alignment_coefficient_kwargs['alpha']['value'] = FLAGS.alpha if FLAGS.alpha is not None else 1.0e-3
            self.alignment_kwargs['alpha'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['alpha'])
            # beta scheduler
            self.alignment_coefficient_kwargs['beta']['value'] = FLAGS.beta if FLAGS.beta is not None else 1.0e-3
            self.alignment_kwargs['beta'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['beta'])
            # build alignment class
            self.params['model_params']['alignment'] = alignment.Mirror(**self.alignment_kwargs)
        elif FLAGS.alignment == 'information':
            print("Using Information Alignment")
            # alpha scheduler
            self.alignment_coefficient_kwargs['alpha']['value'] = FLAGS.alpha if FLAGS.alpha is not None else 2.0e-3
            self.alignment_kwargs['alpha'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['alpha'])
            # beta scheduler
            self.alignment_coefficient_kwargs['beta']['value'] = FLAGS.beta if FLAGS.beta is not None else 1.0e-3
            self.alignment_kwargs['beta'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['beta'])
            # gamma scheduler
            self.alignment_coefficient_kwargs['gamma']['value'] = FLAGS.gamma if FLAGS.gamma is not None else 1.0e-3
            self.alignment_kwargs['gamma'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['gamma'])
            # boolean hyperparameters
            self.alignment_kwargs['reconstruction_reversal'] = FLAGS.reconstruction_reversal
            self.alignment_kwargs['reconstruction_amp'] = FLAGS.reconstruction_amp
            self.alignment_kwargs['use_sparse'] = FLAGS.use_sparse
            # build alignment class
            self.params['model_params']['alignment'] = alignment.Information(**self.alignment_kwargs)
        elif FLAGS.alignment == 'kolen_pollack':
            print("Using Kolen Pollack")
            # alpha scheduler
            self.alignment_coefficient_kwargs['alpha']['value'] = FLAGS.alpha if FLAGS.alpha is not None else 1.0e-3
            self.alignment_kwargs['alpha'] = rate_scheduler.build_alignment_coefficient_schedule(**self.alignment_coefficient_kwargs['alpha'])
            # build alignment class
            self.params['model_params']['alignment'] = alignment.Kolen_Pollack(**self.alignment_kwargs)
        elif FLAGS.alignment is None:
            print("No alignment specified, defaulting to backprop")
        else:
            raise ValueError

        # This is to pass the alignment to the metric_fn, so we can plot the
        # alpha, beta and gamma schedules
        if FLAGS.save_alignment_coefficients:
            print("Saving alignment coefficients :) ")
            self.params['validation_params']['topn_val']['targets'].update({'alignment':self.params['model_params']['alignment']})

        # Caches
        if FLAGS.cache_dir is not None:
            self.params['save_params']['cache_dir'] = '{}/localhost:{}/{}/{}/{}'.format(FLAGS.cache_dir,
                                                                            self.params['save_params']['port'],
                                                                            self.params['save_params']['dbname'],
                                                                            self.params['save_params']['collname'],
                                                                            self.params['save_params']['exp_id'])

        # LR drops
        if FLAGS.manual_lr:
            if FLAGS.rescale_lr:
                scaled_lr = FLAGS.learning_rate * (FLAGS.train_batch_size / (float)(FLAGS.base_batch_size))
            else:
                scaled_lr = FLAGS.learning_rate
            self.lr_scheduler_kwargs = {'scaled_lr': scaled_lr,
                                        'drop': FLAGS.drop,
                                        'boundary_step': FLAGS.boundary_step,
                                        'num_batches_per_epoch': self.NUM_BATCHES_PER_EPOCH,
                                        'warmup_epochs': FLAGS.warmup_epochs}
            self.params['learning_rate_params'] = lr.manual_lr(**self.lr_scheduler_kwargs)

        # Optimizer
        optimizer_params = self._set_optimizer(optimizer_class=FLAGS.optimizer)
        if FLAGS.use_noisy_global_opt:
            print("Using NoisyOptimizer on the global optimizer")
            apply_filter = 'backward' if FLAGS.alignment == 'kolen_pollack' else ''
            if FLAGS.noisy_global_opt_distribution is not None:
                noisy_global_opt = build_noisy_optimizer(optimizer_params['optimizer_class'],
                                                         FLAGS.noisy_global_opt_distribution,
                                                         FLAGS.noisy_global_opt_variance,
                                                         apply_filter=apply_filter)
            else:
                noisy_global_opt = build_noisy_optimizer(optimizer_params['optimizer_class'],
                                                         FLAGS.noisy_opt_distribution,
                                                         FLAGS.noisy_opt_variance,
                                                         apply_filter=apply_filter)
            optimizer_params.update({'optimizer_class': noisy_global_opt})
        self.params['optimizer_params'].update(optimizer_params)

        opt_req_global_step = ['swats', 'radam', 'swrats']
        if FLAGS.alignment_optimizer in opt_req_global_step or \
           FLAGS.optimizer in opt_req_global_step:
           self.params['train_params'].update({'include_global_step': True})

        if FLAGS.alignment_optimizer is not None:
            # have loss returned be a list ([model + reg_loss, alignment losses])
            self.params['loss_params']['agg_func_kwargs']['return_list'] = True
            # pass in optimizer params per optimizer
            alignment_optimizer_params = self._set_optimizer(optimizer_class=FLAGS.alignment_optimizer)
            if FLAGS.use_noisy_alignment_opt:
                print("Using NoisyOptimizer on the alignment optimizer")
                apply_filter = 'backward' if FLAGS.alignment == 'kolen_pollack' else ''
                if FLAGS.noisy_alignment_opt_distribution is not None:
                    noisy_alignment_opt = build_noisy_optimizer(alignment_optimizer_params['optimizer_class'],
                                                                FLAGS.noisy_alignment_opt_distribution,
                                                                FLAGS.noisy_alignment_opt_variance,
                                                                apply_filter=apply_filter)
                else:
                    noisy_alignment_opt = build_noisy_optimizer(alignment_optimizer_params['optimizer_class'],
                                                                FLAGS.noisy_opt_distribution,
                                                                FLAGS.noisy_opt_variance,
                                                                apply_filter=apply_filter)
                alignment_optimizer_params.update({'optimizer_class': noisy_alignment_opt})
            for k in ['optimizer_class', 'optimizer_kwargs']:
                if k == 'optimizer_class':
                    self.params['optimizer_params'][k] = [self.params['optimizer_params'][k], alignment_optimizer_params[k]]
                elif k == 'optimizer_kwargs':
                    model_optimizer_kwargs = self.params['optimizer_params'][k] if k in self.params['optimizer_params'].keys() else {}
                    alignment_optimizer_kwargs = alignment_optimizer_params[k] if k in alignment_optimizer_params.keys() else {}
                    self.params['optimizer_params'][k] = [model_optimizer_kwargs, alignment_optimizer_kwargs]
                else:
                    raise ValueError
            # set learning rate schedule
            if FLAGS.alignment_manual_lr:
                if FLAGS.alignment_rescale_lr:
                    alignment_scaled_lr = FLAGS.alignment_learning_rate * (FLAGS.train_batch_size / (float)(FLAGS.base_batch_size))
                else:
                    alignment_scaled_lr = FLAGS.alignment_learning_rate
                self.alignment_lr_scheduler_kwargs = {
                        'scaled_lr': alignment_scaled_lr,
                        'drop': FLAGS.drop if FLAGS.alignment_drop is None else FLAGS.alignment_drop,
                        'boundary_step': FLAGS.boundary_step if FLAGS.alignment_boundary_step is None else FLAGS.alignment_boundary_step,
                        'num_batches_per_epoch': self.NUM_BATCHES_PER_EPOCH,
                        'warmup_epochs': FLAGS.alignment_warmup_epochs
                    }
                alignment_lr_params = lr.manual_lr(**self.alignment_lr_scheduler_kwargs)
            else:
                self.alignment_lr_scheduler_kwargs = {
                        'learning_rate': FLAGS.alignment_learning_rate,
                        'num_batches_per_epoch': self.NUM_BATCHES_PER_EPOCH,
                        'train_batch_size': FLAGS.train_batch_size,
                        'base_batch_size': FLAGS.base_batch_size,
                        'rescale_lr': FLAGS.alignment_rescale_lr,
                        'constant_lr': FLAGS.alignment_constant_lr,
                        'warmup_epochs': FLAGS.alignment_warmup_epochs,
                        'alternate_step_freq': FLAGS.alternate_step_freq,
                        # delay epochs in rate_scheduler only sets the class loss
                        # rate to 0 so should only apply to categorization to set
                        # its lr to 0 until then delay, but not alignment
                        'delay_epochs': None,
                         # to sync the alignment lr drops with class loss, we
                         # add this to the LR schedule since alignment will be
                         # running for delay epochs longer with nonzero lr
                        'delay_epochs_offset': FLAGS.delay_epochs
                    }
                alignment_lr_params = {'func': lr.build_lr_schedule(**self.alignment_lr_scheduler_kwargs)}

            self.params['learning_rate_params'] = {
                    'func': lr.combined_lr,
                    'lr_params':[self.params['learning_rate_params'], alignment_lr_params]
                }

         # TPU compatibility
        if FLAGS.tpu_name is not None:
            self.make_tpu_compatible(FLAGS)

        # Debugging statements
        print("Saving cache at: {}".format(self.params['save_params']['cache_dir']))

    def make_tpu_compatible(self, flags):
        FLAGS = flags
        if FLAGS.cache_dir is not None:
            tpu_cache_dir = FLAGS.cache_dir
        else:
            tpu_cache_dir = 'neur-al'
        # Update the data params
        self.params['train_params']['data_params'] = {
                    'func': ImageNet(image_dir=FLAGS.data_dir,
                                     prep_type=self._data_prep_type,
                                     is_train=True,
                                     resize=FLAGS.image_size).dataset_func_tpu,
                    'batch_size': FLAGS.train_batch_size}
        self.params['validation_params']['topn_val']['data_params'] = {
                    'func': ImageNet(image_dir=FLAGS.data_dir,
                                     prep_type=self._data_prep_type,
                                     is_train=False,
                                     resize=FLAGS.image_size).dataset_func_tpu,
                    'batch_size': FLAGS.eval_batch_size}
        # Send the model the gpu_mode flag in False
        self.params['model_params']['gpu_mode'] = False
        # Use the tpu version of the loss agg func
        self.params['loss_params']['agg_func'] = losses.mean_loss_with_reg_tpu
        # Change the metric function
        self.params['validation_params']['topn_val']['targets']['func'] = metrics.metric_fn_tpu
        self.params['validation_params']['topn_val'].pop('agg_func')
        self.params['validation_params']['topn_val'].pop('online_agg_func')
        # max checkpoints to keep in gcloud cache
        self.params['save_params']['checkpoint_max'] = FLAGS.checkpoint_max
        # Save caches in glcoud
        self.params['save_params']['cache_dir'] = 'gs://{}/{}/{}/{}/'.format(tpu_cache_dir,
                                                                             self.params['save_params']['dbname'],
                                                                             self.params['save_params']['collname'],
                                                                             self.params['save_params']['exp_id'])

    def save(self, filename):
        # TODO: check if we need to pop the alignment, rate_scheduler, coefficient kwargs
        # and sve them separtately, and recreate the objects when loading
        params_to_save = self.get_params_copy()
        with open(filename, 'wb') as f:
            dill.dump({'params': params_to_save}, f)
            print("Params were saved at {}".format(filename)) #TODO: expand the full path here

    def load(self, filename, flags=None):
        # Parse the initial flags (will set some class attributes)
        if flags is not None:
            self._build_default_params(flags)

        # Load the config
        with open(filename, 'rb') as f:
            saved_params = dill.load(f)
            self.params = saved_params['params']
        # Parse the relevant flags
        if flags is not None:
            self._customize_save(flags)

    def _customize_save(self, flags):
        FLAGS = flags
        # Overwrite save_params
        self.params['save_params'].update({
            'host': 'localhost',
            'port': FLAGS.port,
            'dbname': FLAGS.dbname,
            'cache_dir': FLAGS.cache_dir})
        self.params['save_params']['exp_id'] += FLAGS.exp_id_suffix

        # Allow customizing which device to run on
        self.params['model_params'].update({
            'num_gpus': len(FLAGS.gpu.split(',')),
            'devices': ['/gpu:%i' % idx for idx in range(len(FLAGS.gpu.split(',')))],
            # The following params only needed if tpu, passed as kwargs to model_fn
            # Will only train on TPU is tpu_name is not None
            'tpu_name': FLAGS.tpu_name,
            'gcp_project': FLAGS.gcp_project,
            'tpu_zone': FLAGS.tpu_zone,
            'num_shards': FLAGS.num_shards,
            'iterations_per_loop': FLAGS.iterations_per_loop})

        # Overwrite data params for GPU
        # NOTE: FLAGS.image_size is the only thing not directly saved
        #       in the params dictionary we loaded, so we rely on the user
        #       setting it appropriately
        self.params['train_params']['data_params'] = {
                'func': ImageNet(image_dir=FLAGS.data_dir,
                                 prep_type=self._data_prep_type,
                                 resize=FLAGS.image_size).dataset_func,
                'is_train': True,
                'batch_size': FLAGS.train_batch_size}
        self.params['validation_params']['topn_val']['data_params'] = {
                'func': ImageNet(image_dir=FLAGS.data_dir,
                                 prep_type=self._data_prep_type,
                                 resize=FLAGS.image_size).dataset_func,
                'is_train': False,
                'batch_size': FLAGS.eval_batch_size,
                'q_cap': FLAGS.eval_batch_size,
                'file_pattern': 'validation-*'}

        # Allow for TPU training
        if FLAGS.tpu_name is not None:
            # Make sure the flags take these values from the loaded config file
            FLAGS.train_batch_size = self.params['train_params']['data_params']['batch_size']
            FLAGS.eval_batch_size = self.params['validation_params']['topn_val']['data_params']['batch_size']
            FLAGS.checkpoint_max = self.params['save_params']['checkpoint_max']
            self.make_tpu_compatible(FLAGS)

        # Allow using different types of resnets
        self.params['model_params']['resnet_size'] = (int)(FLAGS.model.split('resnet')[-1])
        self.params['model_params']['use_v2'] = FLAGS.use_resnet_v2
        if FLAGS.use_resnet_v2:
            self.params['save_params']['collname'] += 'v2'

    def get_params_copy(self):
        return copy.deepcopy(self.params)

    def get_alignment_kwargs_copy(self):
        return copy.deepcopy(self.alignment_kwargs)

    def get_rate_scheduler_kwargs_copy(self):
        return copy.deepcopy(self.rate_scheduler_kwargs)

    def get_lr_scheduler_kwargs_copy(self):
        return copy.deepcopy(self.lr_scheduler_kwargs)

    def get_alignment_lr_scheduler_kwargs_copy(self):
        return copy.deepcopy(self.alignment_lr_scheduler_kwargs)

    def get_alignment_coefficient_kwargs_copy(self):
        return copy.deepcopy(self.alignment_coefficient_kwargs)

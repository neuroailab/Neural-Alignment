import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

#### Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either'
         ' this flag or --master.')

flags.DEFINE_integer(
    'num_shards',
    default=8,
    help='Number of shards (TPU cores).')

flags.DEFINE_integer(
    'iterations_per_loop',
    default=100,
    help='Number of interior TPU cycles to run before returning to the host. '
         'This is different from the number of steps run before each eval '
         'and should primarily be used only if you need more incremental '
         'logging during training. Setting this to -1 will set the '
         'iterations_per_loop to be as large as possible (i.e. perform every '
         'call to train in a single TPU loop.')

#### Basic train config params
flags.DEFINE_string(
    'data_dir',
    default=None,
    help='The directory where the ImageNet input data is stored.')

flags.DEFINE_string(
    'cache_dir',
    default=None,
    help='Where the local tfutils cache will be.')

flags.DEFINE_integer(
    'checkpoint_max',
    default=5,
    help='Number of checkpoints to keep in gcloud cache dir (for TPU only).')

flags.DEFINE_string(
    'gpu',
    default='0',
    help='Comma separated list of which gpu indices to use.'
         'For instance, a possible value for this flag is "0,1,5,6" ')

flags.DEFINE_integer(
    'port',
    default=29029,
    help='Port where the MongoDB is accesible at.')

flags.DEFINE_string(
    'dbname',
    default='neur-al',
    help='DBName to be used. Might want to be overridden to keep experiments'
         ' separate')

flags.DEFINE_string(
    'exp_id_suffix',
    default='',
    help='suffix for experiment ID for position saving and tfutils logging')

flags.DEFINE_bool(
    'skip_check',
    default=True,
    help='Whether to skip tfutils version check.')

flags.DEFINE_bool(
    'validate_first',
    default=True,
    help='Whether to validate at first step (turn off when debugging).')

flags.DEFINE_bool(
    'do_restore',
    default=False,
    help='Whether to restore the latest checkpoint from the DB and resume '
         'training')

flags.DEFINE_integer(
    'load_step',
    default=None,
    help='The step to load from. If None, loads the latest checkpoint.')

flags.DEFINE_bool(
    'load_db',
    default=False,
    help='Whether to restore from a different database to continue training.')

flags.DEFINE_integer(
    'load_port',
    default=None,
    help='Port number to load from.')

flags.DEFINE_string(
    'load_dbname',
    default=None,
    help='DBName to be loaded from.')

flags.DEFINE_string(
    'load_collname',
    default=None,
    help='Collname to be loaded from.')

flags.DEFINE_string(
    'load_exp_id',
    default=None,
    help='Exp id to be loaded from.')

flags.DEFINE_string(
    'load_checkpoint',
    default=None,
    help='Checkpoint to load from')

flags.DEFINE_string(
    'val_exp_id',
    default=None,
    help='Exp id to be saved to for validation.')

flags.DEFINE_float(
    'train_epochs',
    default=90,    # Roughly 450412 steps with a batch size of 256
    help='The number of steps to use for training.')

flags.DEFINE_integer(
    'train_batch_size',
    default=256,
    help='Batch size for training.')

flags.DEFINE_integer(
    'minibatch_size',
    default=None,
    help='Batch size for training if you want to minibatch.')

flags.DEFINE_integer(
    'eval_batch_size',
    default=1024,
    help='Batch size for evaluation.')

flags.DEFINE_float(
    'epochs_per_checkpoint',
    default=2,
    help='Controls how often (in epochs) checkpoints are generated.')

flags.DEFINE_string(
    'load_params_file',
    default=None,
    help='Path to a pickle to load the tfutils params dictionary from.'
         'If not set, params will be parsed from the flags as usual. ')

flags.DEFINE_string(
    'save_params_file',
    default=None,
    help='Path to a pickle to save the tfutils params dictionary to. '
         'If not set, will not save the params. ')

flags.DEFINE_string(
    'save_to_gfs',
    default=None,
    help='Comma separated list of keys to save to gfs')

#### Optimizer flags
flags.DEFINE_string(
    'optimizer',
    default='gd',
    help='The optimizer to use for training. Must be one of:\n'
         '* gd\n'
         '* momentum\n'
         '* adagrad\n'
         '* rmsprop\n'
         '* adam\n'
         '* radam\n'
         '* swats\n'
         '* swrats (SWATS with RAdam)\n')

flags.DEFINE_string(
    'alignment_optimizer',
    default=None,
    help='A separate optimizer to use for training the alignment loss. '
         'If None, uses optimizer above. \n'
         'Must be one of:\n'
         '* gd\n'
         '* momentum\n'
         '* adagrad\n'
         '* rmsprop\n'
         '* adam\n'
         '* radam\n'
         '* swats\n'
         '* swrats (SWATS with RAdam)\n')

flags.DEFINE_bool(
    'use_noisy_global_opt',
    default=False,
    help='Whether to wrap the global optimizer in the noisy optimizer')

flags.DEFINE_bool(
    'use_noisy_alignment_opt',
    default=False,
    help='Whether to wrap the alignment optimizer in the noisy optimizer')

flags.DEFINE_string(
    'noisy_global_opt_distribution',
    default=None,
    help='Name of the distribution for noisy global optimizer to use. '
         'Must be None (use layer input) or one of:\n'
         '* uniform\n'
         '* normal\n')

flags.DEFINE_float(
    'noisy_global_opt_variance',
    default=1.0,
    help='Variance of noisy global optimizer distribution if one is given\n')

flags.DEFINE_string(
    'noisy_alignment_opt_distribution',
    default=None,
    help='Name of the distribution for noisy alignment optimizer to use. '
         'Must be None (use layer input) or one of:\n'
         '* uniform\n'
         '* normal\n')

flags.DEFINE_float(
    'noisy_alignment_opt_variance',
    default=1.0,
    help='Variance of noisy alignment optimizer distribution if one is given\n')

flags.DEFINE_string(
    'noisy_opt_distribution',
    default=None,
    help='Name of the distribution for all noisy optimizers to use. '
         'Must be None (use layer input) or one of:\n'
         '* uniform\n'
         '* normal\n')

flags.DEFINE_float(
    'noisy_opt_variance',
    default=1.0,
    help='Variance of all noisy optimizers distribution if one is given\n')

flags.DEFINE_bool(
    'grad_clip',
    default=False,
    help='Whether to clip gradients.')

flags.DEFINE_float(
    'grad_clipping_value',
    default=1.0,
    help='Gradient clipping value')

flags.DEFINE_string(
    'grad_clipping_method', default='norm',
    help='How to clip. [value | norm]')

#### Learning rate flags
flags.DEFINE_float(
    'learning_rate',
    default=0.001,
    help='base learning assuming a batch size of 256.'
         'For other batch sizes it is scaled linearly with batch size.')

flags.DEFINE_float(
    'alignment_learning_rate',
    default=0.001,
    help='If using separate alignment optimizer, base alignment loss learning'
         ' assuming a batch size of 256.'
         'For other batch sizes it is scaled linearly with batch size.')

flags.DEFINE_bool(
    'rescale_lr',
    default=False,
    help='whether to rescale the LR')

flags.DEFINE_bool(
    'alignment_rescale_lr',
    default=False,
    help='whether to rescale the alignment loss')

flags.DEFINE_bool(
    'constant_lr',
    default=True,
    help='whether to have a constant learning rate')

flags.DEFINE_bool(
    'alignment_constant_lr',
    default=True,
    help='whether to have a constant learning rate for the alignment loss')

flags.DEFINE_integer(
    'base_batch_size',
    default=256,
    help='base batch size to rescale learning rate by.'
         'For other batch sizes it is scaled linearly with batch size.')

flags.DEFINE_bool(
    'manual_lr',
    default=False,
    help='whether to manually drop the LR')

flags.DEFINE_bool(
    'alignment_manual_lr',
    default=False,
    help='whether to manually drop the LR for the alignment loss')

flags.DEFINE_float(
    'warmup_epochs',
    default=5,
    help='Number of epochs to do learning rate warmup '
         '(defaults to 5 if not specified).')

flags.DEFINE_float(
    'alignment_warmup_epochs',
    default=5,
    help='Number of epochs to do learning rate warmup for alignment loss '
         '(defaults to 5 if not specified).')

flags.DEFINE_integer(
    'drop',
    default=0,
    help='The current learning rate drop index')

flags.DEFINE_integer(
    'alignment_drop',
    default=None,
    help='The current learning rate drop index for the alignment loss. '
         'If None, it is synced with the categorization loss drop flag.')

flags.DEFINE_integer(
    'boundary_step',
    default=None,
    help='The step from which the drop in the larning rate should start')

flags.DEFINE_integer(
    'alignment_boundary_step',
    default=None,
    help='The step from which the drop in the larning rate should start for '
         'the alignment loss')

##### Dataset flags
flags.DEFINE_string(
    'dataset',
    default='mnist',
    help='Name of the dataset to use for training. Must be one of:\n'
         '* mnist\n'
         '* imagenet\n')

flags.DEFINE_integer(
    'image_size',
    default=None,
    help='Image size for ImageNet. '
         'Default is None, which means 224x224 sized images')

#### Model flags
flags.DEFINE_string(
    'model',
    default='fc',
    help='Name of the model architecture to use. Must be one of:\n'
         '* fc \n'
         '* resnet18 \n')

flags.DEFINE_bool(
    'use_resnet_v2',
    default=False,
    help='Whether to use resnet v2 models. (Default is to use v1)')

flags.DEFINE_bool(
    'tf_layers',
    default=False,
    help='Whether to override our custom layers with tf_layers.'
         'For debugging purposes only. Should be equivalent to alignment=None')

flags.DEFINE_string(
    'layers_list',
    default=None,
    help='Comma separated list of number of units in hidden layers')

flags.DEFINE_string(
    'activation',
    default='sigmoid',
    help='Activation used for hidden layers in model on of:\n'
         '* sigmoid\n'
         '* tanh\n'
         '* relu\n')

#### Alignment flags
flags.DEFINE_string(
    'alignment',
    default=None,
    help='Name of the alignment to use. Must be None (Backprop) or one of:\n'
         '* feedback\n'
         '* symmetric\n'
         '* activation\n'
         '* mirror\n'
         '* information\n'
         '* kolen_pollack\n')

flags.DEFINE_bool(
    'save_alignment_coefficients',
    default=False,
    help='Whether to save the alpha, beta and gamma coefficients during '
         'train-time validations (only on GPU)')

flags.DEFINE_float(
    'alpha',
    default=None,
    help='Alpha weight of the alignment')

flags.DEFINE_float(
    'alpha_start',
    default=0.0,
    help='Fraction (between 0 and 1) of training to start non-zero alpha weight')

flags.DEFINE_float(
    'alpha_stop',
    default=1.0,
    help='Fraction (between 0 and 1) of training to end non-zero alpha weight')

flags.DEFINE_float(
    'alpha_cycle',
    default=None,
    help='Number of epochs for triangular cycle period')

flags.DEFINE_float(
    'alpha_schedule_rate',
    default=0.0,
    help='Rate for growth or decay of alpha weight in non-zero regime')

flags.DEFINE_string(
    'alpha_schedule_type',
    default=None,
    help='Type of schedule for alpha weight.  One of \n'
         '* linear\n'
         '* exponential\n'
         '* cyclic\n')

flags.DEFINE_float(
    'beta',
    default=None,
    help='Beta weight of the alignment')

flags.DEFINE_float(
    'beta_start',
    default=0.0,
    help='Fraction (between 0 and 1) of training to start non-zero beta weight')

flags.DEFINE_float(
    'beta_stop',
    default=1.0,
    help='Fraction (between 0 and 1) of training to end non-zero beta weight')

flags.DEFINE_float(
    'beta_cycle',
    default=None,
    help='Number of epochs for triangular cycle period')

flags.DEFINE_float(
    'beta_schedule_rate',
    default=0.0,
    help='Rate for growth or decay of beta weight in non-zero regime')

flags.DEFINE_string(
    'beta_schedule_type',
    default=None,
    help='Type of schedule for beta weight.  One of \n'
         '* linear\n'
         '* exponential\n'
         '* cyclic\n')

flags.DEFINE_float(
    'gamma',
    default=None,
    help='Gamma weight of the alignment (so far only used for Information '
         'Alignment)')

flags.DEFINE_float(
    'gamma_start',
    default=0.0,
    help='Fraction (between 0 and 1) of training to start non-zero gamma weight')

flags.DEFINE_float(
    'gamma_stop',
    default=1.0,
    help='Fraction (between 0 and 1) of training to end non-zero gamma weight')

flags.DEFINE_float(
    'gamma_cycle',
    default=None,
    help='Number of epochs for triangular cycle period')

flags.DEFINE_float(
    'gamma_schedule_rate',
    default=0.0,
    help='Rate for growth or decay of gamma weight in non-zero regime')

flags.DEFINE_string(
    'gamma_schedule_type',
    default=None,
    help='Type of schedule for gamma weight.  One of \n'
         '* linear\n'
         '* exponential\n'
         '* cyclic\n')

flags.DEFINE_bool(
    'reconstruction_reversal',
    default=False,
    help='Whether to use the backwards reconstruction instead of forward'
         ' reconstruction for the "null" primitive')

flags.DEFINE_bool(
    'reconstruction_amp',
    default=False,
    help='Whether to use input and reconstruction in "amp" primitive'
         ' instead of forward and backward projection')

flags.DEFINE_bool(
    'use_sparse',
    default=False,
    help='Whether to use the sparse primitive instead of decay'
         ' primnitive for information alignment')

flags.DEFINE_bool(
    'update_forward',
    default=False,
    help='Boolean defining whether alignment loss will modify the forward weights')

flags.DEFINE_string(
    'input_distribution',
    default=None,
    help='Name of the input distribution to use. '
         'Must be None (use layer input) or one of:\n'
             '* uniform\n'
             '* normal\n')

flags.DEFINE_float(
    'input_stddev',
    default=1.0,
    help='If using a non-None input distribution, then uses this value as the '
         'standard deviation. (Defaults to 1.0)')

flags.DEFINE_bool(
    'use_bias_forward',
    default=True,
    help='Boolean defining whether to use biases in forward layers')

flags.DEFINE_bool(
    'use_bias_backward',
    default=True,
    help='Boolean defining whether to use biases in backward layers')

flags.DEFINE_string(
    'activation_fn_override',
    default=None,
    help='Activation to override default in alignment regularization computations:\n'
         '* sigmoid\n'
         '* tanh\n'
         '* relu\n')

flags.DEFINE_bool(
    'activation_forward',
    default=False,
    help='Boolean defining whether alignment loss should be calculated using '
         'the activations')

flags.DEFINE_bool(
    'activation_backward',
    default=False,
    help='Boolean defining whether alignment loss should be calculated using '
         'the activations')

flags.DEFINE_bool(
    'bn_trainable',
    default=True,
    help='Boolean defining whether batch normalization variables should be '
         'trainable')

flags.DEFINE_bool(
    'batch_center_backward_input',
    default=False,
    help='Boolean defining whether to batch center the backward inputs to layers')

flags.DEFINE_bool(
    'center_input',
    default=False,
    help='Boolean defining whether to center the forward inputs to layers')

flags.DEFINE_bool(
    'normalize_input',
    default=False,
    help='Boolean defining whether to normalize the forward inputs to layers')

flags.DEFINE_bool(
    'batch_center_forward_output',
    default=False,
    help='Boolean defining whether to batch center the forward outputs of layers')

flags.DEFINE_bool(
    'center_forward_output',
    default=False,
    help='Boolean defining whether to center the forward outputs of layers')

flags.DEFINE_bool(
    'normalize_forward_output',
    default=False,
    help='Boolean defining whether to normalize the forward outputs of layers')

flags.DEFINE_bool(
    'center_backward_output',
    default=False,
    help='Boolean defining whether to center the backwards outputs of layers')

flags.DEFINE_bool(
    'normalize_backward_output',
    default=False,
    help='Boolean defining whether to normalize the backwards outputs of layers')

flags.DEFINE_float(
    'loss_rate',
    default=1.0,
    help='Loss weighting (categorization + reg)')

flags.DEFINE_float(
    'alignment_rate',
    default=1.0,
    help='Alignment loss weighting')

flags.DEFINE_float(
    'delay_epochs',
    default=None,
    help='Float defining number of epochs to delay applying loss update and '
         'only apply alignment update')

flags.DEFINE_integer(
    'alternate_step_freq',
    default=None,
    help='Integer defining number of steps (must be at least 1) to apply loss '
         'update per alignment update')

flags.DEFINE_bool(
    'constant_rate',
    default=True,
    help='Boolean defining whether to have a constant rate or apply a schedule')

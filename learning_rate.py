import tensorflow as tf
import copy

LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
        (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

# This is used in the default params
def build_lr_schedule(learning_rate=0.1,
                      num_batches_per_epoch=1,
                      train_batch_size=128,
                      base_batch_size=256,
                      rescale_lr=False,
                      constant_lr=True,
                      warmup_epochs=None,
                      alternate_step_freq=None,
                      delay_epochs=None,
                      delay_epochs_offset=None):
    def learning_rate_schedule(global_step):
        """Handles linear scaling rule, gradual warmup, and LR decay.
        The learning rate starts at 0, then it increases linearly per step.
        After 5 epochs we reach the base learning rate (scaled to account
        for batch size).
        After 30, 60 and 80 epochs the learning rate is divided by 10.
        Args:
        current_epoch: `Tensor` for current epoch.
        Returns:
        A scaled `Tensor` for current learning rate.
        """

        lr_sched = copy.deepcopy(LR_SCHEDULE)
        current_epoch = (tf.cast(global_step, tf.float32) / num_batches_per_epoch)

        if delay_epochs is not None:
            # this is usually used for the class loss
            current_epoch = tf.where(tf.greater_equal(current_epoch, delay_epochs),
                                     current_epoch - delay_epochs,
                                     0.0)

        if warmup_epochs is not None:
            # we overwrite the lr_sched so that start_epoch is updated down below
            lr_sched[0] = (lr_sched[0][0], warmup_epochs)

        if alternate_step_freq is not None:
            # we amplify the number of epochs
            assert(isinstance(alternate_step_freq, int))
            assert(alternate_step_freq >= 1)
            for lr_idx, lr_elem in enumerate(lr_sched):
                lr_elem_alt = (lr_elem[0], lr_elem[1] * (alternate_step_freq + 1))
                lr_sched[lr_idx] = lr_elem_alt

        if delay_epochs_offset is not None:
            # use one or the other, this is usually used for the alignment loss
            # to make it sync with lr drops of class loss
            assert(delay_epochs is None)
            for lr_idx, lr_elem in enumerate(lr_sched):
              if lr_idx > 0:
                  lr_elem_offset = (lr_elem[0], lr_elem[1] + delay_epochs_offset)
              else:
                  # we explicitly set the warmup epochs to be 0 because the
                  # whole idea is to use a non zero learning rate on the
                  # alignment loss if you are using delay epochs
                  lr_elem_offset = (lr_elem[0], 0)
              lr_sched[lr_idx] = lr_elem_offset

        if rescale_lr:
            scaled_lr = learning_rate * (train_batch_size / (float)(base_batch_size))
        else:
            scaled_lr = learning_rate

        if constant_lr:
            return tf.constant(scaled_lr)
        else:
            print('Using this LR schedule', lr_sched)
            if lr_sched[0][1] > 0:
                decay_rate = (scaled_lr * lr_sched[0][0] *
                              current_epoch / lr_sched[0][1])
            else:
                decay_rate = scaled_lr * lr_sched[0][0]

            for mult, start_epoch in lr_sched:
                decay_rate = tf.where(current_epoch < start_epoch,
                                      decay_rate, scaled_lr * mult)
            return decay_rate
    return learning_rate_schedule

# This will perform the LR drop, if manual LR drops are used
def drop_learning_rate(global_step,
                       boundary_step,
                       next_lr,
                       prev_lr=None):
    """ Handles a single learning rate drop, as specified by the params
    """
    assert boundary_step is not None, ("You didn't specify a boundary step")
    if prev_lr is None:
            prev_lr = 10.0 * next_lr
    lr = tf.train.piecewise_constant(x=global_step,
                                     boundaries=[boundary_step],
                                     values=[prev_lr, next_lr])
    return lr

def warmup_learning_rate(global_step,
                         target_lr,
                         num_batches_per_epoch,
                         warmup_epochs):

    current_epoch = (tf.cast(global_step, tf.float32) / num_batches_per_epoch)
    decay_rate = target_lr * current_epoch / warmup_epochs
    decay_rate = tf.where(current_epoch < warmup_epochs, decay_rate, target_lr)
    return decay_rate

def manual_lr(scaled_lr,
              drop,
              boundary_step,
              num_batches_per_epoch=1,
              warmup_epochs=None):
    # Default values from the training script
    params_dict = {
        'func':tf.train.exponential_decay,
        'learning_rate':0.01,
        'decay_steps':1e5,
        'decay_rate':1.0}
    if drop == 0:
        if warmup_epochs is None:
            params_dict['learning_rate_params']['learning_rate'] = scaled_lr
        else:
            params_dict = {'func': warmup_learning_rate,
                           'target_lr': scaled_lr,
                           'num_batches_per_epoch': num_batches_per_epoch,
                           'warmup_epochs': warmup_epochs}
    elif drop > 0:
        prev_lr = ((0.1)**(drop-1))*(scaled_lr)
        next_lr = ((0.1)**drop)*(scaled_lr)
        print('Prev_lr:', prev_lr,
              'next_lr:', next_lr,
              'boundary_step:', boundary_step)
        params_dict = {'func': drop_learning_rate,
                                               'boundary_step': boundary_step,
                                               'next_lr': next_lr,
                                               'prev_lr': prev_lr}

    return params_dict

def combined_lr(global_step, lr_params):
    '''
    This function combines multiple learning rate parameters, useful for
    multiple optimizer training
    lr_params is a list of dictionaries, with the func and kwargs
    '''
    assert(isinstance(lr_params, list) is True)
    lr_list = []
    for p in lr_params:
        curr_params = copy.deepcopy(p)
        curr_func = curr_params.pop('func')
        curr_lr = curr_func(global_step=global_step, **curr_params)
        lr_list.append(curr_lr)

    return lr_list

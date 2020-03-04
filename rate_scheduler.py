import tensorflow as tf
import numpy as np
from Metrics.losses import get_global_step_var

RT_SCHEDULE = [    # (multiplier, epoch to start) tuples
        (1.0, 1.0, 5), (0.1, 0.1, 30), (0.01, 0.01, 60), (0.001, 0.001, 80)
]

def build_schedule(loss_rate=1.0,
                   alignment_rate=1.0,
                   delay_epochs=None,
                   alternate_step_freq=None,
                   constant_rate=True,
                   num_batches_per_epoch=1):
    def rate_schedule(global_step):
        step = tf.cast(global_step, tf.float32)
        epoch = step / num_batches_per_epoch

        # constant rates
        l_rate = tf.constant(loss_rate)
        a_rate = tf.constant(alignment_rate)

        # delay loss rate
        if delay_epochs is not None:
            l_rate = tf.where(epoch < delay_epochs, 0.0, l_rate)
            # If you wanted to magnify the aligment during the delay, you could
            # do something like this:
            # a_rate = tf.where(epoch < delay_epochs, 1e3, a_rate)

        # alternate loss and alignment rates
        if alternate_step_freq is not None:
            assert(isinstance(alternate_step_freq, int))
            assert(alternate_step_freq >= 1)
            mod = tf.mod(step, (alternate_step_freq + 1))
            l_rate = tf.where(tf.not_equal(mod, 0), l_rate, 0.0)
            if delay_epochs is not None:
                a_rate = tf.where(tf.logical_or(tf.equal(mod, 0),
                                                epoch < delay_epochs),
                                  a_rate,
                                  0.0)
            else:
                a_rate = tf.where(tf.equal(mod, 0), a_rate, 0.0)

        # decay rates
        if not constant_rate:
            l_rate_ = tf.identity(l_rate)
            a_rate_ = tf.identity(a_rate)
            for loss_mult, alignment_mult, start in RT_SCHEDULE:
                l_rate_ = tf.where(epoch < start, l_rate_, l_rate * loss_mult)
                a_rate_ = tf.where(epoch < start, a_rate_, a_rate * alignment_mult)
            l_rate = l_rate_
            a_rate = a_rate_

        return l_rate, a_rate

    return rate_schedule

def triangular_cycle(current_epoch,
                     stepsize_in_epochs):
    current_cycle = tf.floor(1 + tf.divide(current_epoch, 2.0 * stepsize_in_epochs))
    current_x = tf.abs(tf.divide(current_epoch, stepsize_in_epochs) - 2.0 * current_cycle + 1)
    current_pos = tf.maximum(0.0, 1.0 - current_x)
    return current_pos

def build_alignment_coefficient_schedule(value=1.0,
                                         start=0.0,
                                         stop=1.0,
                                         cycle=0,
                                         schedule_rate=0.0,
                                         schedule_type=None,
                                         num_batches_per_epoch=1,
                                         train_epochs=1):
    def alignment_coefficient_schedule():
        try:
            global_step = get_global_step_var()
        except:
            # This is really only used in test_from_params mode, when the
            #  global step does not exist, but we do not want this to crash
            global_step = tf.Variable(0, name='global_step')
        step = tf.cast(global_step, tf.float32)
        epoch = step / num_batches_per_epoch

        # constant rates
        rate = tf.constant(value)

        # start/stop steps
        start_epoch = start * train_epochs
        stop_epoch = stop * train_epochs

        # current epoch
        curr_epoch = epoch - start_epoch

        # compute change
        if schedule_type is not None:
            if cycle > 0:
                stepsize_in_epochs = 0.5 * cycle
                current_pos = triangular_cycle(current_epoch=curr_epoch,
                                         stepsize_in_epochs=stepsize_in_epochs)
            else:
                curr_epoch = epoch - start_epoch
                current_pos = curr_epoch
                stepsize_in_epochs = 1.0
            if schedule_type == 'linear':
                rate += (schedule_rate * current_pos * stepsize_in_epochs)
            elif schedule_type == 'exponential':
                rate *= tf.math.exp(schedule_rate * current_pos * stepsize_in_epochs)
            else:
                raise ValueError

        # threshold
        rate = tf.where(epoch < start_epoch, 0.0, rate)
        rate = tf.where(stop_epoch < epoch, 0.0, rate)

        return rate

    return alignment_coefficient_schedule

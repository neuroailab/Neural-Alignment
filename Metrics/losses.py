import tensorflow as tf
from tfutils.multi_gpu.easy_variable_mgr import COPY_NAME_SCOPE

def category_loss(logits, labels, **kwargs):
    '''
    Computes category loss
    '''
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                          logits=logits)
    return loss

def get_global_step_var():
    global_step_vars = [v for v in tf.global_variables() \
                        if 'global_step' in v.name]
    assert len(global_step_vars) == 1
    return global_step_vars[0]

# Provide multi-gpu safe regularization loss computation.
# Used as agg_func in loss_params
def filter_losses(losses, which_device):
    loss = tf.constant(0.0)
    curr_name_scope = '%s%i' % (COPY_NAME_SCOPE, which_device)
    valid_losses = filter(
            lambda v: curr_name_scope in v.name,
            losses)
    if len(valid_losses) > 0:
        loss += tf.add_n(valid_losses)
    return loss

def loss_metric(inputs, outputs, target, **kwargs):
    report_dict = {}
    model_loss = tf.reduce_mean(category_loss(logits=outputs,
                                              labels=inputs[target]))
    report_dict['model_loss'] = model_loss
    # model regularization
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Adding the losses, for now, since we could not filter
    # them out per GPU
    reg_loss = tf.add_n(reg_losses)
    report_dict['reg_loss'] = reg_loss
    # alignment regularization
    alignment_losses = tf.get_collection('ALIGNMENT_LOSSES')
    # Adding the losses, for now, since we could not filter
    # them out per GPU
    if len(alignment_losses) > 0:
        alignment_loss = tf.add_n(alignment_losses)
    else:
        alignment_loss = tf.constant(0.0)
    report_dict['alignment_loss'] = alignment_loss
    # apply weighted combination of model, reg, and alignment losses
    global_step = get_global_step_var()
    default_rate = lambda x: (tf.constant(1.0), tf.constant(1.0))
    rate = kwargs.get('rate', default_rate)
    loss_rate, alignment_rate = rate(global_step)
    loss = loss_rate*(model_loss + reg_loss) + alignment_rate*alignment_loss
    report_dict['total_loss'] = loss
    return report_dict

def _combine_losses(model_loss, reg_loss, alignment_loss, **kwargs):
    # apply weighted combination of model, reg, and alignment losses
    global_step = get_global_step_var()
    default_rate = lambda x: (tf.constant(1.0), tf.constant(1.0))
    rate = kwargs.get('rate', default_rate)
    loss_rate, alignment_rate = rate(global_step)
    # get total losses
    total_model_loss = loss_rate*(model_loss + reg_loss)
    total_alignment_loss = alignment_rate*alignment_loss
    return_list = kwargs.get('return_list', False)
    if return_list:
        loss = [total_model_loss, total_alignment_loss]
    else:
        loss = total_model_loss + total_alignment_loss
    return loss

def mean_loss_with_reg(loss, which_device, **kwargs):
    """
    takes batch mean of model loss and adds model regularization and alignment
    regularization
    """
    model_loss = tf.reduce_mean(loss)
    # model regularization
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = filter_losses(reg_losses, which_device)
    # alignment regularization
    alignment_losses = tf.get_collection('ALIGNMENT_LOSSES')
    alignment_loss = filter_losses(alignment_losses, which_device)
    loss = _combine_losses(model_loss, reg_loss, alignment_loss, **kwargs)
    return loss

def mean_loss_with_reg_tpu(loss, **kwargs):
    """
    takes batch mean of model loss and adds model regularization and alignment
    regularization
    """
    model_loss = tf.reduce_mean(loss)
    # model regularization
    reg_loss =  tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # alignment regularization
    # Since we don't use filter_losses (like on GPU), explicitly set it to zero
    if len(tf.get_collection('ALIGNMENT_LOSSES')) == 0:
        alignment_loss = tf.zeros_like(model_loss)
    else:
        alignment_loss = tf.add_n(tf.get_collection('ALIGNMENT_LOSSES'))
    loss = _combine_losses(model_loss, reg_loss, alignment_loss, **kwargs)
    return loss

def mean_loss_noreg(loss):
    return tf.reduce_mean(loss)

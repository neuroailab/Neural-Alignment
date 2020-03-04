import tensorflow as tf
import numpy as np
import losses
import utils

def index_of(item, coll):
    index = 0
    while item not in coll[index] and index < len(coll):
        index += 1
    return index

def nicename(name):
    name_parts = name.split("/")
    if 'block_group' in name:
        i = index_of('block_group', name_parts)
        name = "/".join(name_parts[i:i+2])
    elif 'dense' in name:
        i = index_of('dense', name_parts)
        name = name_parts[i]
    elif 'conv' in name:
        i = index_of('conv', name_parts)
        name = name_parts[i]
    return name

def var_grab_fn(k):
    return {nicename(t.name): t for op in tf.get_default_graph().get_operations() \
                                for t in op.values() \
                                if (k in t.name and 'validation' not in t.name)}

def metric_fn(inputs, outputs, num_classes,
              alignment=None,
              include_global_step=True):
    # On GPU inputs is a dictionary, and outputs is a tensor
    logits = outputs
    labels = inputs['labels']

    precision_at_1 = utils.accuracy(logits, labels, 1)
    precision_at_5 = utils.accuracy(logits, labels, 5)

    # category_loss = losses.build_category_loss(num_classes=num_classes)
    # loss = category_loss(logits=logits, labels=labels)
    # mean_loss_with_reg = losses.mean_loss_noreg(loss)
    # alignment_loss = tf.get_collection('ALIGNMENT_LOSSES')

    metrics_dict = {
        'top1': precision_at_1,
        'top5': precision_at_5,
        #'loss': loss,
        'weight_angles': var_grab_fn('weight_angle'),
        'projection_angles': var_grab_fn('projection_angle'),
        'gradient_angles': var_grab_fn('gradient_angle'),
        # 'mean_loss_with_reg': mean_loss_with_reg,
        # 'alignment_losses': alignment_loss
    }
    metrics_dict['hebb_angle'] = var_grab_fn('hebb_angle')
    for k1 in ['weightsq', 'actsq']:
        for k2 in ['norm', 'mean']:#, 'var']:
            k = k1 + '_' + k2
            metrics_dict[k] = var_grab_fn(k)

    if include_global_step:
        metrics_dict['global_step'] = losses.get_global_step_var()

    if alignment is not None:
        coeffs = {}
        if hasattr(alignment, 'alpha'):
            coeffs['alpha'] = alignment.alpha()
        if hasattr(alignment, 'beta'):
            coeffs['beta'] = alignment.beta()
        if hasattr(alignment, 'gamma'):
            coeffs['gamma'] = alignment.gamma()
        metrics_dict['alignment_coefficients'] = coeffs

    return metrics_dict

def metric_fn_tpu(labels, logits):
    # The TPU interface is a bit more constrained: logits and labels are tensors
    # and requires accuracy to be
    # calculated exactly like this. utils.accuracy will not work on TPU
    predictions = tf.argmax(logits, axis=1)
    precision_at_1 = tf.metrics.accuracy(labels, predictions)

    in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
    precision_at_5 = tf.metrics.mean(in_top_5)

    return {
        'top1': precision_at_1,
        'top5': precision_at_5,
    }

def online_agg_append(agg_res, res, step):
    # This funciton aggregates within a batch
    # This is run outside of tensorflow. A batch is evaluated, and the
    # results of metric_fn on the batch are fed in here.
    # We then aggregate them somehow In this case,
    # just concat the results from the metric_fn, if it is not
    # the loss, and the accuracies
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.iteritems():
        if k in ['loss', 'mean_loss_with_reg', 'top1', 'top5', 'global_step']:
            agg_res[k].append(np.mean(v))
        else:
            agg_res[k].append(v)
    return agg_res

def concat_agg_func(results):
    # Now this function aggregates across batches, and does the same thing as
    # before just concat the results from the online_agg_append func, if it is
    # not the loss, and the accuracies
    agg_res = {}
    for k, v in results.iteritems():
        if k in ['loss', 'mean_loss_with_reg', 'top1', 'top5', 'global_step']:
            agg_res[k] = np.mean(v)
        elif k in ['weight_angles', 'projection_angles', 'gradient_angles',
                   'weightsq_norm', 'weightsq_mean', 'actsq_norm', 'actsq_mean',
                   'hebb_angle', 'alignment_coefficients']:
            agg_res[k] = {}
            # v is a list of dictionaries
            for key in v[0].keys():
                values = np.array([d[key] for d in v])
                agg_res[k][key] = np.mean(values)
        else:
            agg_res[k] = np.concatenate(v, axis=0)
    return agg_res

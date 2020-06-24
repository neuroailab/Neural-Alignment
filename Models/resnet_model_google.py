from resnet_model_v1 import resnet_v1
from resnet_model_v2 import resnet_v2

from collections import OrderedDict

#### This is the tfutils wrapper around the Google resnet
def google_resnet_func(inputs,
                       train=True,
                       resnet_size=50,
                       num_classes=1000,
                       alignment=None,
                       tf_layers=False,
                       use_v2=False,
                       gpu_mode=True,
                       return_endpoints=False,
                       bn_trainable=True,
                       regularize_weights=True,
                       **kwargs):

    params = OrderedDict()
    if tf_layers:
        print("Using tf.layers")
    else:
        print("Using custom layers")
    if use_v2:
        print("using v2 resnet-", resnet_size, " as model")
        network = resnet_v2(
                resnet_size=resnet_size, num_classes=num_classes,
                alignment=alignment, tf_layers=tf_layers,
                bn_trainable=bn_trainable,
                regularize_weights=regularize_weights)
    else:
        print("using v1 resnet-", resnet_size, " as model")
        network = resnet_v1(
                resnet_depth=resnet_size, num_classes=num_classes,
                alignment=alignment, tf_layers=tf_layers,
                bn_trainable=bn_trainable,
                regularize_weights=regularize_weights)
    print("gpu_mode set to: {}".format(gpu_mode))
    if gpu_mode:
        print('inputs shape', inputs['images'].shape)
        logits = network(
                inputs=inputs['images'], is_training=train,
                return_endpoints=return_endpoints)
        if return_endpoints:
            logits, endpoints = logits
            return logits, endpoints, params
        else:
            print('Logits shape', logits.shape)
            return logits, params
    else:
        # TPU mode
        print('inputs shape', inputs.shape)
        logits = network(
                inputs=inputs, is_training=train)
        print('Logits shape', logits.shape)
        return logits

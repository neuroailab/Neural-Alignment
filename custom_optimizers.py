# import relevant tensorflow ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import numpy as np

def _if(cond, true_op, false_op):
    """Helper function to replicate if/else control flow with tf.cond
    This is more readable than the statement below
    Assumes cond is a boolean tensor
    """
    #return tf.cond(cond, true_fn=lambda: true_op, false_fn=lambda: false_op)
    return tf.where(cond, true_op, false_op)

def _if_num(cond, true_op, false_op):
    """Helper function to replicate if/else control flow with a numeric
    condition. The condition is assumed to be 0 or 1 and the same dtype as the
    two argument ops
    """
    return cond*true_op + (1-cond)*false_op

class SWATSOptimizer(optimizer.Optimizer):
    """Implementation of SWATS,
    from:https://arxiv.org/pdf/1712.07628.pdf
    @@__init__
    """
    def __init__(self,
                 learning_rate=0.001,
                 global_step=None,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-09,
                 rectified_adam=False,
                 use_locking=False,
                 name="SWATS"):

        super(SWATSOptimizer, self).__init__(use_locking, name)

        if global_step is None:
            self.gs = tf.train.get_global_step()
        else:
            self.gs = global_step

        self._lr = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._rectified_adam = rectified_adam

        # do RADAM: Algorithm 2 of https://arxiv.org/pdf/1908.03265v1.pdf
        if self._rectified_adam:
            self._rho_inf = (2.0/(1.0 - self._beta_2)) - 1.0
            self._rho_inf_t = None

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._gs_t = None
        self._beta_1_t = None
        self._beta_2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._gs_t = ops.convert_to_tensor(self.gs, name="gs_t")
        self._beta_1_t = ops.convert_to_tensor(self._beta_1, name="beta_1_t")
        self._beta_2_t = ops.convert_to_tensor(self._beta_2, name="beta_2_t")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon_t")
        if self._rectified_adam:
            self._rho_inf_t = ops.convert_to_tensor(self._rho_inf, name="rho_inf_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            # 0 phase means Adam, > 0 phase means SGDM
            self._zeros_slot(v, "phase", self._name)
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "momentum", self._name)
            self._zeros_slot(v, "Lambda", self._name) # resultant SGDM lr
            self._zeros_slot(v, "p_k", self._name)

    def _apply_dense(self, grad, var):
        # cast input tensors to variable dtype
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self._beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self._beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        gs_t = math_ops.cast(self._gs_t, var.dtype.base_dtype)
        if self._rectified_adam:
            rho_inf_t = math_ops.cast(self._rho_inf_t, var.dtype.base_dtype)

        # get slot variables
        phase = self.get_slot(var, "phase")
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        momentum = self.get_slot(var, "momentum")
        Lambda = self.get_slot(var, "Lambda")
        p_k = self.get_slot(var, "p_k")

        phase_adam = tf.equal(phase, tf.constant(0.0, dtype=var.dtype.base_dtype))

        m_adam = beta_1_t * m + (1.0 - beta_1_t) * grad
        m_sgdm = m
        m_t = _if(phase_adam, m_adam, m_sgdm)

        v_adam = beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad)
        v_sgdm = v
        v_t = _if(phase_adam, v_adam, v_sgdm)

        beta_1_power = tf.pow(beta_1_t, gs_t + 1.0)
        beta_2_power = tf.pow(beta_2_t, gs_t + 1.0)
        term_1 = tf.divide(tf.sqrt(1.0 - beta_2_power), (1.0 - beta_1_power))
        term_2 = tf.divide(m_t, tf.sqrt(v_t) + epsilon_t)
        if self._rectified_adam:
            rho_k_t = rho_inf_t - tf.divide((2.0*(gs_t+1.0)*beta_2_power),
                                            (1.0 - beta_2_power))
            rect_cond = tf.greater(rho_k_t, tf.constant(5.0, dtype=var.dtype.base_dtype))
            r_num = (rho_k_t - 4.0) * (rho_k_t - 2.0) * (rho_inf_t)
            r_denom = (rho_inf_t - 4.0) * (rho_inf_t - 2.0) * (rho_k_t)
            r_t = tf.sqrt(tf.divide(r_num, r_denom))
            p_k_adam_rect = -1.0 * lr_t * r_t * term_1 * term_2
            p_k_adam_uncorrected = -1.0 * lr_t * tf.divide(m_t,
                                                           1.0 - beta_1_power)
            p_k_adam = _if(rect_cond, p_k_adam_rect, p_k_adam_uncorrected)
        else:
            p_k_adam = -1.0 * lr_t * term_1 * term_2

        p_k_sgdm = p_k
        p_k_t = _if(phase_adam, p_k_adam, p_k_sgdm)

        p_k_flat = tf.reshape(p_k_t, [-1])
        grad_flat = tf.reshape(grad, [-1])
        inner_prod = tf.tensordot(p_k_flat, grad_flat, axes=1)
        inner_prod_cond = tf.not_equal(inner_prod, tf.constant(0.0, dtype=var.dtype.base_dtype))
        p_k_norm = tf.tensordot(p_k_flat, p_k_flat, axes=1)
        gamma = -1.0 * tf.divide(p_k_norm, inner_prod)
        Lambda_adam = Lambda * beta_2_t + (1.0 - beta_2_t) * gamma
        Lambda_sgdm = Lambda
        Lambda_cond = tf.logical_and(phase_adam, inner_prod_cond)
        Lambda_int = _if(Lambda_cond, Lambda_adam, Lambda_sgdm)

        abs_cond = tf.less(tf.abs(tf.divide(Lambda_int,
                                            1.0 - beta_2_power) - gamma),
                           epsilon_t)
        switch_cond = tf.logical_and(tf.greater(gs_t + 1.0, 1.0), abs_cond)
        phase_cond = tf.logical_and(tf.logical_and(phase_adam, inner_prod_cond), switch_cond)
        phase_t = _if(phase_cond, tf.ones_like(phase, dtype=var.dtype.base_dtype), phase)
        # learning rate for SGDM
        Lambda_t = _if(phase_cond, tf.divide(Lambda_int, 1.0 - beta_2_power), Lambda_int)

        # var update
        momentum_t = _if(phase_adam, momentum, beta_1_t*momentum + grad)
        adam_var_update = -1.0*p_k_t
        sgdm_var_update = (1.0 - beta_1_t)*Lambda_t*momentum_t
        var_update = _if(phase_adam, adam_var_update, sgdm_var_update)

        # all var and slot variable update ops
        var_update_op = state_ops.assign_sub(var, var_update)
        momentum_update_op = momentum.assign(momentum_t)
        Lambda_update_op = Lambda.assign(Lambda_t)
        phase_update_op = phase.assign(phase_t)
        p_k_update_op = p_k.assign(p_k_t)
        v_update_op = v.assign(v_t)
        m_update_op = m.assign(m_t)

        all_update_ops = [var_update_op, momentum_update_op, Lambda_update_op, \
                          phase_update_op, p_k_update_op, v_update_op,
                          m_update_op]

        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*all_update_ops)

    # CAUTION: uncomment this for use with TPU training, not sure if this is correct
    # def _resource_apply_dense(self, grad, var):
    #     return self._apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


class RAdamOptimizer(optimizer.Optimizer):
    """Implementation of RAdam, can be more useful than LR warmup
    from: Algorithm 2 of https://arxiv.org/pdf/1908.03265v1.pdf
    @@__init__
    """
    def __init__(self,
                 learning_rate=0.001,
                 global_step=None,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-09,
                 use_locking=False,
                 name="RAdam"):

        super(RAdamOptimizer, self).__init__(use_locking, name)

        if global_step is None:
            self.gs = tf.train.get_global_step()
        else:
            self.gs = global_step

        self._lr = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._rho_inf = (2.0/(1.0 - self._beta_2)) - 1.0
        self._rho_inf_t = None

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._gs_t = None
        self._beta_1_t = None
        self._beta_2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._gs_t = ops.convert_to_tensor(self.gs, name="gs_t")
        self._beta_1_t = ops.convert_to_tensor(self._beta_1, name="beta_1_t")
        self._beta_2_t = ops.convert_to_tensor(self._beta_2, name="beta_2_t")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon_t")
        self._rho_inf_t = ops.convert_to_tensor(self._rho_inf, name="rho_inf_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        # cast input tensors to variable dtype
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self._beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self._beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        gs_t = math_ops.cast(self._gs_t, var.dtype.base_dtype)
        rho_inf_t = math_ops.cast(self._rho_inf_t, var.dtype.base_dtype)

        # get slot variables
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = beta_1_t * m + (1.0 - beta_1_t) * grad
        v_t = beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad)

        beta_1_power = tf.pow(beta_1_t, gs_t + 1.0)
        beta_2_power = tf.pow(beta_2_t, gs_t + 1.0)
        term_1 = tf.divide(tf.sqrt(1.0 - beta_2_power), (1.0 - beta_1_power))
        term_2 = tf.divide(m_t, tf.sqrt(v_t) + epsilon_t)
        rho_k_t = rho_inf_t - tf.divide((2.0*(gs_t+1.0)*beta_2_power),
                                        (1.0 - beta_2_power))
        # This "relaxed" condition is not in the paper, but it is in their code
        # https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py#L132
        rect_cond = tf.greater_equal(rho_k_t, tf.constant(5.0, dtype=var.dtype.base_dtype))
        r_num = (rho_k_t - 4.0) * (rho_k_t - 2.0) * (rho_inf_t)
        r_denom = (rho_inf_t - 4.0) * (rho_inf_t - 2.0) * (rho_k_t)
        r_t = tf.sqrt(tf.divide(r_num, r_denom))
        p_k_adam_rect = -1.0 * lr_t * r_t * term_1 * term_2
        p_k_adam_uncorrected = -1.0 * lr_t * (tf.divide(m_t, 1.0 - beta_1_power))
        p_k_t = _if(rect_cond, p_k_adam_rect, p_k_adam_uncorrected)

        # var update
        # NOTE: we do not need to keep track of p_k like above in SWATS since
        # we are not switching to keep it the same in SGD
        var_update = -1.0*p_k_t

        # all var and slot variable update ops
        var_update_op = state_ops.assign_sub(var, var_update)
        v_update_op = v.assign(v_t)
        m_update_op = m.assign(m_t)

        all_update_ops = [var_update_op, v_update_op, m_update_op]

        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*all_update_ops)

    # CAUTION: uncomment this for use with TPU training, not sure if this is correct
    # def _resource_apply_dense(self, grad, var):
    #     return self._apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


def build_noisy_optimizer(optimizer_class,
                          distribution,
                          variance,
                          apply_filter=''):
    class NoisyOptimizer(optimizer_class):
        def __init__(self,
                     **optimizer_kwargs):
            super(NoisyOptimizer, self).__init__(**optimizer_kwargs)
            self.distribution = distribution
            self.variance = variance
            self._lr = optimizer_kwargs['learning_rate']

        def _noise(self, var):
            if self.distribution == 'uniform':
                val = np.sqrt(12 * self.variance) / 2
                noise = tf.random.uniform(tf.shape(var), minval=-val, maxval=val)
            elif self.distribution == 'normal':
                noise = tf.random.normal(tf.shape(var), stddev=np.sqrt(self.variance))
            else:
                raise ValueError
            return noise

        def _apply_dense(self, grad, var):
            update_op = super(NoisyOptimizer, self)._apply_dense(grad, var)
            all_update_ops = [update_op]
            assert type(apply_filter) is str
            if apply_filter in var.name:
                noise_op = var.assign_add(self._lr * self._noise(var))
                all_update_ops.append(noise_op)

            return control_flow_ops.group(*all_update_ops)

        def _resource_apply_dense(self, grad, var):
            update_op = super(NoisyOptimizer, self)._resource_apply_dense(grad, var)
            all_update_ops = [update_op]
            assert type(apply_filter) is str
            if apply_filter in var.name:
                noise_op = var.assign_add(self._lr * self._noise(var))
                all_update_ops.append(noise_op)

            return control_flow_ops.group(*all_update_ops)

    return NoisyOptimizer

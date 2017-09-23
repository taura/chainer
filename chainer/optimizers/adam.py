import math

import numpy

from chainer import cuda
from chainer import optimizer

tau_opt=1
if tau_opt:
    import os
    import chopt
    from chopt import c_long,c_int,c_float,a_int,a_float,a_float2d
    tau_chk=0

_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.alpha = 0.001
_default_hyperparam.beta1 = 0.9
_default_hyperparam.beta2 = 0.999
_default_hyperparam.eps = 1e-8


class AdamRule(optimizer.UpdateRule):

    """Update rule of Adam optimization algorithm.

    See :class:`~chainer.optimizers.Adam` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.Hyperparameter): Hyperparameter that
            provides the default values.
        alpha (float): Step size.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, parent_hyperparam=None,
                 alpha=None, beta1=None, beta2=None, eps=None):
        super(AdamRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if beta1 is not None:
            self.hyperparam.beta1 = beta1
        if beta2 is not None:
            self.hyperparam.beta2 = beta2
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            self.state['m'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        if tau_opt:
            m,v,data = self.update_core_cpu_new(param)
            if tau_chk:
                self.update_core_cpu_org(param)
                chopt.check_array_error(m, self.state['m'])
                chopt.check_array_error(v, self.state['v'])
                chopt.check_array_error(data, param.data)
                self.state['m'][...] = m
                self.state['v'][...] = v
                param.data[...] = data
        else:
            self.update_core_cpu_org(param)

    def update_core_cpu_new(self, param):
        grad = param.grad       # 9999,100
        m, v = self.state['m'], self.state['v'] # 9999,100
        data = param.data
        if grad is None:
            return m,v,data
        if tau_chk:
            m = m.copy()
            v = v.copy()
            data = data.copy()
        hp = self.hyperparam
        beta1 = hp.beta1
        beta2 = hp.beta2
        eps = hp.eps
        lr = self.lr
        assert(grad.shape == m.shape), (grad.shape, m.shape)
        assert(grad.shape == v.shape), (grad.shape, v.shape)
        assert(grad.shape == param.data.shape), (grad.shape, param.data.shape)
        assert(grad.dtype == numpy.float32), grad.dtype
        assert(m.dtype == numpy.float32), m.dtype
        assert(v.dtype == numpy.float32), v.dtype
        assert(data.dtype == numpy.float32), data.dtype
        M,N = grad.shape
        update = chopt.make_fun("update", "libadam_c.so",
                                [c_long]*2 + [c_float]*4 + [a_float2d]*4,
                                c_int)
        update(M, N, beta1, beta2, lr, eps, grad, m, v, data)
        return m,v,data

    def update_core_cpu_org(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        m, v = self.state['m'], self.state['v']

        m += (1 - hp.beta1) * (grad - m)
        v += (1 - hp.beta2) * (grad * grad - v)
        param.data -= self.lr * m / (numpy.sqrt(v) + hp.eps)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
            'T param, T m, T v',
            '''m += one_minus_beta1 * (grad - m);
               v += one_minus_beta2 * (grad * grad - v);
               param -= lr * m / (sqrt(v) + eps);''',
            'adam')(grad, self.lr, 1 - self.hyperparam.beta1,
                    1 - self.hyperparam.beta2, self.hyperparam.eps, param.data,
                    self.state['m'], self.state['v'])

    @property
    def lr(self):
        fix1 = 1. - self.hyperparam.beta1 ** self.t
        fix2 = 1. - self.hyperparam.beta2 ** self.t
        return self.hyperparam.alpha * math.sqrt(fix2) / fix1


class Adam(optimizer.GradientMethod):

    """Adam optimizer.

    See: http://arxiv.org/abs/1412.6980v8

    Args:
        alpha (float): Step size.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self,
                 alpha=_default_hyperparam.alpha,
                 beta1=_default_hyperparam.beta1,
                 beta2=_default_hyperparam.beta2,
                 eps=_default_hyperparam.eps):
        super(Adam, self).__init__()
        self.hyperparam.alpha = alpha
        self.hyperparam.beta1 = beta1
        self.hyperparam.beta2 = beta2
        self.hyperparam.eps = eps

    alpha = optimizer.HyperparameterProxy('alpha')
    beta1 = optimizer.HyperparameterProxy('beta1')
    beta2 = optimizer.HyperparameterProxy('beta2')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return AdamRule(self.hyperparam)

    @property
    def lr(self):
        fix1 = 1. - self.hyperparam.beta1 ** self.t
        fix2 = 1. - self.hyperparam.beta2 ** self.t
        return self.hyperparam.alpha * math.sqrt(fix2) / fix1

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check

tau_opt=1
if tau_opt:
    import chopt
    from chopt import c_long,c_int,a_int,a_int2d,a_float,a_float2d,a_float3d
    import os

class EmbedIDFunction(function.Function):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types
        type_check.expect(
            x_type.dtype == numpy.int32,
            x_type.ndim >= 1,
        )
        type_check.expect(
            w_type.dtype == numpy.float32,
            w_type.ndim == 2
        )

    def forward(self, inputs):
        x, W = inputs

        xp = cuda.get_array_module(*inputs)
        if chainer.is_debug():
            valid_x = xp.logical_and(0 <= x, x < len(W))
            if self.ignore_label is not None:
                valid_x = xp.logical_or(valid_x, x == self.ignore_label)
            if not valid_x.all():
                raise ValueError('Each not ignored `x` value need to satisfy'
                                 '`0 <= x < len(W)`')

        if self.ignore_label is not None:
            mask = (x == self.ignore_label)
            return xp.where(
                mask[..., None], 0, W.take(xp.where(mask, 0, x), axis=0)),

        return W.take(x, axis=0),

    def backward(self, inputs, grad_outputs):
        if tau_opt:
            none_n,gW_n = self.backward_new(inputs, grad_outputs)
            if 0:
                none_o,gW_o = self.backward_org(inputs, grad_outputs)
                err = ((gW_n - gW_o)**2).sum()
                assert(err <= 1.0e-10 * (gW_o**2).sum()), err
            return none_n,gW_n
        else:
            return self.backward_org(inputs, grad_outputs)

    def backward_new(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, W = inputs           # x: (1000,2), W: (10000, 100)
        gy = grad_outputs[0]    # (1000, 2, 100)
        gW = xp.zeros_like(W)   # (10000, 100)

        if xp is numpy:
            # It is equivalent to `numpy.add.at(gW, x, gy)` but ufunc.at is
            # too slow.
            # x.ravel()               : (2000,)
            # gy.reshape(x.size, -1)  : (2000, 100)
            M,N = x.shape
            K,L = W.shape
            assert(gy.shape == (M, N, L)), (gy.shape, x.shape, W.shape)
            ignore_label = self.ignore_label
            if ignore_label is None:
                ignore_label = -1
            backward = chopt.make_fun("backward", "libembed_id_c.so",
                                      [c_long]*5 + [a_int2d,a_float2d,a_float3d],
                                      c_int)
            backward(M, N, K, L, ignore_label, x, gW, gy)
        else:
            if self.ignore_label is None:
                cuda.elementwise(
                    'T gy, int32 x, int32 n_out', 'raw T gW',
                    'int w_ind[] = {x, i % n_out}; atomicAdd(&gW[w_ind], gy)',
                    'embed_id_bwd')(
                        gy, xp.expand_dims(x, -1), gW.shape[1], gW)
            else:
                cuda.elementwise(
                    'T gy, int32 x, int32 n_out, int32 ignore', 'raw T gW',
                    '''
                    if (x != ignore) {
                      int w_ind[] = {x, i % n_out};
                      atomicAdd(&gW[w_ind], gy);
                    }
                    ''',
                    'embed_id_bwd_ignore_label')(
                        gy, xp.expand_dims(x, -1), gW.shape[1],
                        self.ignore_label, gW)
        return None, gW

    def backward_org(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, W = inputs
        gy = grad_outputs[0]
        gW = xp.zeros_like(W)

        if xp is numpy:
            # It is equivalent to `numpy.add.at(gW, x, gy)` but ufunc.at is
            # too slow.
            for ix, igy in six.moves.zip(x.ravel(),
                                         gy.reshape(x.size, -1)):
                if ix == self.ignore_label:
                    continue
                gW[ix] += igy
        else:
            if self.ignore_label is None:
                cuda.elementwise(
                    'T gy, int32 x, int32 n_out', 'raw T gW',
                    'int w_ind[] = {x, i % n_out}; atomicAdd(&gW[w_ind], gy)',
                    'embed_id_bwd')(
                        gy, xp.expand_dims(x, -1), gW.shape[1], gW)
            else:
                cuda.elementwise(
                    'T gy, int32 x, int32 n_out, int32 ignore', 'raw T gW',
                    '''
                    if (x != ignore) {
                      int w_ind[] = {x, i % n_out};
                      atomicAdd(&gW[w_ind], gy);
                    }
                    ''',
                    'embed_id_bwd_ignore_label')(
                        gy, xp.expand_dims(x, -1), gW.shape[1],
                        self.ignore_label, gW)
        return None, gW


def embed_id(x, W, ignore_label=None):
    """Efficient linear function for one-hot input.

    This function implements so called *word embedding*. It takes two
    arguments: a set of IDs (words) ``x`` in :math:`B` dimensional integer
    vector, and a set of all ID (word) embeddings ``W`` in :math:`V \\times d`
    float32 matrix. It outputs :math:`B \\times d` matrix whose ``i``-th
    column is the ``x[i]``-th column of ``W``.

    This function is only differentiable on the input ``W``.

    Args:
        x (~chainer.Variable): Batch vectors of IDs.
        W (~chainer.Variable): Representation of each ID (a.k.a.
            word embeddings).
        ignore_label (int or None): If ``ignore_label`` is an int value,
            ``i``-th column of return value is filled with ``0``.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.EmbedID`

    """
    return EmbedIDFunction(ignore_label=ignore_label)(x, W)

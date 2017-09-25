import copy

import numpy
import six

from chainer import cuda
from chainer import function
from chainer import link
from chainer.utils import type_check

tau_opt=1
if tau_opt:
    import chopt
    from chopt import c_long,c_int,c_float,c_double,a_int,a_float,a_float2d
    import os
    import pdb
    # tau_ok = int(os.environ["OK"])

class TreeParser(object):

    def __init__(self):
        self.next_id = 0

    def size(self):
        return self.next_id

    def get_paths(self):
        return self.paths

    def get_codes(self):
        return self.codes

    def parse(self, tree):
        self.next_id = 0
        self.path = []
        self.code = []
        self.paths = {}
        self.codes = {}
        self._parse(tree)

        assert(len(self.path) == 0)
        assert(len(self.code) == 0)
        assert(len(self.paths) == len(self.codes))

    def _parse(self, node):
        if isinstance(node, tuple):
            # internal node
            if len(node) != 2:
                raise ValueError(
                    'All internal nodes must have two child nodes')
            left, right = node
            self.path.append(self.next_id)
            self.next_id += 1
            self.code.append(1.0)
            self._parse(left)

            self.code[-1] = -1.0
            self._parse(right)

            self.path.pop()
            self.code.pop()

        else:
            # leaf node
            self.paths[node] = numpy.array(self.path, dtype=numpy.int32)
            self.codes[node] = numpy.array(self.code, dtype=numpy.float32)

class BinaryHierarchicalSoftmaxFunction(function.Function):

    """Hierarchical softmax function based on a binary tree.

    This function object should be allocated beforehand, and be copied on every
    forward computation, since the initializer parses the given tree. See the
    implementation of :class:`BinaryHierarchicalSoftmax` for details.

    Args:
        tree: A binary tree made with tuples like ``((1, 2), 3)``.

    .. seealso::
       See :class:`BinaryHierarchicalSoftmax` for details.

    """

    def __init__(self, tree):
        parser = TreeParser()
        parser.parse(tree)
        paths = parser.get_paths()
        codes = parser.get_codes()
        n_vocab = max(paths.keys()) + 1

        self.paths = numpy.concatenate(
            [paths[i] for i in range(n_vocab) if i in paths])
        self.codes = numpy.concatenate(
            [codes[i] for i in range(n_vocab) if i in codes])
        begins = numpy.empty((n_vocab + 1,), dtype=numpy.int32)
        begins[0] = 0
        for i in range(0, n_vocab):
            length = len(paths[i]) if i in paths else 0
            begins[i + 1] = begins[i] + length
        self.begins = begins

        self.parser_size = parser.size()

        if tau_opt:
            max_length = chopt.make_fun("max_length",
                                        "libhierarchical_softmax_c.so",
                                        [c_long, c_long, a_int, a_int, c_int], c_int)
            forward = chopt.make_fun("forward", 
                                     "libhierarchical_softmax_c.so",
                                     [c_long]*6 + [a_float2d, a_int, a_float2d,
                                                   a_int, a_int, a_float,
                                                   a_float2d],
                                     c_float)
            backward = chopt.make_fun("backward", "libhierarchical_softmax_c.so",
                                      [c_long]*6 + [c_float, a_float2d, a_int,
                                                    a_float2d, a_float2d, a_float2d,
                                                    a_int, a_int, a_float,
                                                    a_float2d ],
                                      c_int)
            self.fun_max_length = max_length
            self.fun_forward = forward
            self.fun_backward = backward
        
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, t_type, w_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
            w_type.dtype == numpy.float32,
            w_type.ndim == 2,
            w_type.shape[0] == self.parser_size,
            w_type.shape[1] == x_type.shape[1],
        )

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            self.paths = cuda.to_gpu(self.paths)
            self.codes = cuda.to_gpu(self.codes)
            self.begins = cuda.to_gpu(self.begins)

    def to_cpu(self):
        self.paths = cuda.to_cpu(self.paths)
        self.codes = cuda.to_cpu(self.codes)
        self.begins = cuda.to_cpu(self.begins)

    def forward_cpu(self, inputs):
        if tau_opt:
            n = self.forward_cpu_new(inputs)
            if 0:               # check
                o = self.forward_cpu_org(inputs)
                chopt.check_array_error(n[0], o[0])
            return n
        else:
            return self.forward_cpu_org(inputs)
        
    def forward_cpu_new(self, inputs):
        x, t, W = inputs
        begins = self.begins
        paths = self.paths
        codes = self.codes
        M,N = x.shape
        P,_ = W.shape
        bn, = begins.shape
        pn, = paths.shape
        if 0:
            assert(t.shape == (M,)), (t.shape, x.shape)
            assert(W.shape == (P,N)), (W.shape, x.shape)
            assert(codes.shape == (pn,)), (codes.shape, paths.shape)
            assert(x.dtype == numpy.float32), x.dtype
            assert(t.dtype == numpy.int32), t.dtype
            assert(W.dtype == numpy.float32), W.dtype
            assert(begins.dtype == numpy.int32), begins.dtype
            assert(paths.dtype == numpy.int32), paths.dtype
            assert(codes.dtype == numpy.float32), codes.dtype
            max_length = chopt.make_fun("max_length",
                                        "libhierarchical_softmax_c.so",
                                        [c_long, c_long, a_int, a_int, c_int], c_int)
            forward = chopt.make_fun("forward", 
                                     "libhierarchical_softmax_c.so",
                                     [c_long]*6 + [a_float2d, a_int, a_float2d,
                                                   a_int, a_int, a_float,
                                                   a_float2d],
                                     c_float)
        ml = self.fun_max_length(bn, M, begins, t, 0)
        wxy = numpy.zeros((M, ml), dtype=numpy.float32)
        loss = self.fun_forward(M, N, P, bn, pn, ml, x, t, W, begins, paths, codes, wxy)
        self.wxy = wxy
        return numpy.array(loss),

    def xx_forward_cpu_new(self, inputs):
        x, t, W = inputs
        ml = 0
        for ix, it in six.moves.zip(x, t):
            ml = max(ml, self.begins[it + 1] - self.begins[it])
        wxy = numpy.zeros((x.shape[0], ml), dtype=numpy.float32)
        
        loss = numpy.float32(0.0)
        dot = chopt.make_fun("dot", "libhierarchical_softmax_c.so",
                             [ c_long, a_float, a_float ], c_double)
        for i,(ix, it) in enumerate(six.moves.zip(x, t)):
            begin = self.begins[it]
            end = self.begins[it + 1]

            w = W[self.paths[begin:end]]
            for k in range(begin, end):
                o = k - begin
                wix = w[o].dot(ix)
                nn, = w[o].shape
                my_wix = dot(nn, w[o], ix)
                chopt.check_scal_error(wix, my_wix)
                wxy[i,o] = wix * self.codes[k]
            l = numpy.logaddexp(0.0, -wxy[i,:end-begin]) # == log(1 + exp(-wxy))
            loss += numpy.sum(l)
        self.wxy = wxy
        return numpy.array(loss),

    def forward_cpu_org(self, inputs):
        x, t, W = inputs

        loss = numpy.float32(0.0)
        for ix, it in six.moves.zip(x, t):
            loss += self._forward_cpu_one(ix, it, W)
        return numpy.array(loss),

    def _forward_cpu_one(self, x, t, W):
        begin = self.begins[t]
        end = self.begins[t + 1]

        w = W[self.paths[begin:end]]
        wxy = w.dot(x) * self.codes[begin:end]
        loss = numpy.logaddexp(0.0, -wxy)  # == log(1 + exp(-wxy))
        return numpy.sum(loss)

    def backward_cpu(self, inputs, grad_outputs):
        if tau_opt:
            gx_n,none_n,gW_n = self.backward_cpu_new(inputs, grad_outputs)
            if 0:               # check
                gx_o,none_o,gW_o = self.backward_cpu_org(inputs, grad_outputs)
                chopt.check_array_error(gx_n, gx_o)
                chopt.check_array_error(gW_n, gW_o)
            return gx_n,none_n,gW_n
        else:
            return self.backward_cpu_org(inputs, grad_outputs)

    def x_backward_cpu_new(self, inputs, grad_outputs):
        x, t, W = inputs
        gloss, = grad_outputs
        gx = numpy.empty_like(x)
        gW = numpy.zeros_like(W)
        for i, (ix, it) in enumerate(six.moves.zip(x, t)):
            begin = self.begins[it]
            end = self.begins[it + 1]
            assert(self.wxy.shape[1] >= end - begin), (self.wxy.shape, begin, end)
            path = self.paths[begin:end]
            w = W[path]
            wxy_ = w.dot(ix) * self.codes[begin:end]
            wxy = self.wxy[i,:end-begin]
            chopt.check_array_error(wxy, wxy_)
            g = -gloss * self.codes[begin:end] / (1.0 + numpy.exp(wxy))
            gx[i] = g.dot(w)
            gw = g.reshape((g.shape[0], 1)).dot(ix.reshape(1, ix.shape[0]))
            gW[path] += gw
        return gx, None, gW

    # OK
    def ok_backward_cpu_new(self, inputs, grad_outputs):
        x, t, W = inputs
        M,N = x.shape
        P,_ = W.shape
        gloss, = grad_outputs
        gx = numpy.zeros_like(x)
        gW = numpy.zeros_like(W)
        for i in range(x.shape[0]):
            ix = x[i]           # N
            it = t[i]
            begin = self.begins[it]
            end = self.begins[it + 1]
            assert(self.wxy.shape[1] >= end - begin), (self.wxy.shape, begin, end)
            # print("i: %d, it: %d, begin: %d, end: %d" % (i, it, begin, end))
            for o,k in enumerate(range(begin, end)):
                p = self.paths[k]
                w = W[p]        # N
                wxy_ = w.dot(ix) * self.codes[k]
                wxy = self.wxy[i,o]
                chopt.check_scal_error(wxy, wxy_)
                g = (-gloss * self.codes[k]) / (1.0 + numpy.exp(wxy))
                for j in range(N):
                    if j % (N // 10) == -1:
                        print("gl: %.18f" % gloss)
                        print("codes[%d]: %.18f" % (k, self.codes[k]))
                        print("-gl * codes[%d]: %.18f" % (k, (-gloss * self.codes[k])))
                        print("wxy_c: %.18f" % wxy)
                        print("expf(wxy_c): %.18f" % numpy.exp(wxy))
                        print("1.0 + expf(wxy_c): %.18f" % (1.0 + numpy.exp(wxy)))
                        print("(-gl * codes[k]) / (1.0 + expf(wxy_c)): %.18f" % ((-gloss * self.codes[k]) / (1.0 + numpy.exp(wxy))))
                        print("g: %.18f" % g)
                        print("gx[%d,%d] (%.18f) += %.18f * W[%d,%d] (%.18f)" % (i, j, gx[i,j], g, p, j, W[p,j]))
                        print("gW[%d,%d] (%.18f) += %.18f * x[%d,%d] (%.18f)" % (p, j, gW[i,j], g, i, j, x[i,j]))
                    gx[i,j] += g * W[p,j]  # N
                    gW[p,j] += g * x[i,j] # N
        return gx, None, gW

    # NG
    def backward_cpu_new(self, inputs, grad_outputs):
        x, t, W = inputs
        gloss, = grad_outputs
        begins = self.begins
        paths = self.paths
        codes = self.codes
        wxy = self.wxy
        gx = numpy.zeros_like(x)
        gW = numpy.zeros_like(W)
        M,N = x.shape
        P,_ = W.shape
        bn, = begins.shape
        pn, = paths.shape
        _,ml = wxy.shape
        if 0:
            assert(t.shape == (M,)), (t.shape, x.shape)
            assert(W.shape == (P,N)), (W.shape, x.shape)
            assert(codes.shape == (pn,)), (codes.shape, paths.shape)
            assert(wxy.shape == (M, ml)), (wxy.shape, x.shape)
            assert(x.dtype == numpy.float32), x.dtype
            assert(t.dtype == numpy.int32), t.dtype
            assert(W.dtype == numpy.float32), W.dtype
            assert(begins.dtype == numpy.int32), begins.dtype
            assert(paths.dtype == numpy.int32), paths.dtype
            assert(codes.dtype == numpy.float32), codes.dtype
            backward = chopt.make_fun("backward", "libhierarchical_softmax_c.so",
                                      [c_long]*6 + [c_float, a_float2d, a_int,
                                                    a_float2d, a_float2d, a_float2d,
                                                    a_int, a_int, a_float,
                                                    a_float2d ],
                                      c_int)
        self.fun_backward(M, N, P, bn, pn, ml, gloss,
                          x, t, W, gx, gW, begins, paths, codes, wxy)
        return gx, None, gW
    
    def backward_cpu_org(self, inputs, grad_outputs):
        x, t, W = inputs
        gloss, = grad_outputs
        gx = numpy.empty_like(x)
        gW = numpy.zeros_like(W)
        for i, (ix, it) in enumerate(six.moves.zip(x, t)):
            gx[i] = self._backward_cpu_one(ix, it, W, gloss, gW)
        return gx, None, gW

    def _backward_cpu_one(self, x, t, W, gloss, gW):
        begin = self.begins[t]
        end = self.begins[t + 1]

        path = self.paths[begin:end]
        w = W[path]
        wxy = w.dot(x) * self.codes[begin:end]
        g = -gloss * self.codes[begin:end] / (1.0 + numpy.exp(wxy))
        gx = g.dot(w)
        gw = g.reshape((g.shape[0], 1)).dot(x.reshape(1, x.shape[0]))
        gW[path] += gw
        return gx

    def forward_gpu(self, inputs):
        x, t, W = inputs
        max_length = cuda.reduce(
            'T t, raw T begins', 'T out', 'begins[t + 1] - begins[t]',
            'max(a, b)', 'out = a', '0',
            'binary_hierarchical_softmax_max_length')(t, self.begins)
        max_length = cuda.to_cpu(max_length)[()]

        length = max_length * x.shape[0]
        ls = cuda.cupy.empty((length,), dtype=numpy.float32)
        n_in = x.shape[1]
        wxy = cuda.cupy.empty_like(ls)
        cuda.elementwise(
            '''raw T x, raw T w, raw int32 ts, raw int32 paths,
            raw T codes, raw int32 begins, int32 c, int32 max_length''',
            'T ls, T wxy',
            '''
            int ind = i / max_length;
            int offset = i - ind * max_length;
            int t = ts[ind];

            int begin = begins[t];
            int length = begins[t + 1] - begins[t];

            if (offset < length) {
              int p = begin + offset;
              int node = paths[p];

              T wx = 0;
              for (int j = 0; j < c; ++j) {
                int w_ind[] = {node, j};
                int x_ind[] = {ind, j};
                wx += w[w_ind] * x[x_ind];
              }
              wxy = wx * codes[p];
              ls = log(1 + exp(-wxy));
            } else {
              ls = 0;
            }
            ''',
            'binary_hierarchical_softmax_forward'
        )(x, W, t, self.paths, self.codes, self.begins, n_in, max_length, ls,
          wxy)
        self.max_length = max_length
        self.wxy = wxy
        return ls.sum(),

    def backward_gpu(self, inputs, grad_outputs):
        x, t, W = inputs
        gloss, = grad_outputs

        n_in = x.shape[1]
        gx = cuda.cupy.zeros_like(x)
        gW = cuda.cupy.zeros_like(W)
        cuda.elementwise(
            '''T wxy, raw T x, raw T w, raw int32 ts, raw int32 paths,
            raw T codes, raw int32 begins, raw T gloss,
            int32 c, int32 max_length''',
            'raw T gx, raw T gw',
            '''
            int ind = i / max_length;
            int offset = i - ind * max_length;
            int t = ts[ind];

            int begin = begins[t];
            int length = begins[t + 1] - begins[t];

            if (offset < length) {
              int p = begin + offset;
              int node = paths[p];
              T code = codes[p];

              T g = -gloss[0] * code / (1.0 + exp(wxy));
              for (int j = 0; j < c; ++j) {
                int w_ind[] = {node, j};
                int x_ind[] = {ind, j};
                atomicAdd(&gx[x_ind], g * w[w_ind]);
                atomicAdd(&gw[w_ind], g * x[x_ind]);
              }
            }
            ''',
            'binary_hierarchical_softmax_bwd'
        )(self.wxy, x, W, t, self.paths, self.codes, self.begins, gloss, n_in,
          self.max_length, gx, gW)
        return gx, None, gW


class BinaryHierarchicalSoftmax(link.Link):

    """Hierarchical softmax layer over binary tree.

    In natural language applications, vocabulary size is too large to use
    softmax loss.
    Instead, the hierarchical softmax uses product of sigmoid functions.
    It costs only :math:`O(\\log(n))` time where :math:`n` is the vocabulary
    size in average.

    At first a user need to prepare a binary tree whose each leaf is
    corresponding to a word in a vocabulary.
    When a word :math:`x` is given, exactly one path from the root of the tree
    to the leaf of the word exists.
    Let :math:`\\mbox{path}(x) = ((e_1, b_1), \\dots, (e_m, b_m))` be the path
    of :math:`x`, where :math:`e_i` is an index of :math:`i`-th internal node,
    and :math:`b_i \\in \\{-1, 1\\}` indicates direction to move at
    :math:`i`-th internal node (-1 is left, and 1 is right).
    Then, the probability of :math:`x` is given as below:

    .. math::

       P(x) &= \\prod_{(e_i, b_i) \\in \\mbox{path}(x)}P(b_i | e_i)  \\\\
            &= \\prod_{(e_i, b_i) \\in \\mbox{path}(x)}\\sigma(b_i x^\\top
               w_{e_i}),

    where :math:`\\sigma(\\cdot)` is a sigmoid function, and :math:`w` is a
    weight matrix.

    This function costs :math:`O(\\log(n))` time as an average length of paths
    is :math:`O(\\log(n))`, and :math:`O(n)` memory as the number of internal
    nodes equals :math:`n - 1`.

    Args:
        in_size (int): Dimension of input vectors.
        tree: A binary tree made with tuples like `((1, 2), 3)`.

    Attributes:
        W (~chainer.Variable): Weight parameter matrix.

    See: Hierarchical Probabilistic Neural Network Language Model [Morin+,
    AISTAT2005].

    """

    def __init__(self, in_size, tree):
        # This function object is copied on every forward computation.
        self._func = BinaryHierarchicalSoftmaxFunction(tree)
        super(BinaryHierarchicalSoftmax, self).__init__(
            W=(self._func.parser_size, in_size))
        self.W.data[...] = numpy.random.uniform(-1, 1, self.W.shape)

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            super(BinaryHierarchicalSoftmax, self).to_gpu(device)
            self._func.to_gpu(device)

    def to_cpu(self):
        super(BinaryHierarchicalSoftmax, self).to_cpu()
        self._func.to_cpu()

    @staticmethod
    def create_huffman_tree(word_counts):
        """Makes a Huffman tree from a dictionary containing word counts.

        This method creates a binary Huffman tree, that is required for
        :class:`BinaryHierarchicalSoftmax`.
        For example, ``{0: 8, 1: 5, 2: 6, 3: 4}`` is converted to
        ``((3, 1), (2, 0))``.

        Args:
            word_counts (dict of int key and int or float values):
                Dictionary representing counts of words.

        Returns:
            Binary Huffman tree with tuples and keys of ``word_coutns``.

        """
        if len(word_counts) == 0:
            raise ValueError('Empty vocabulary')

        q = six.moves.queue.PriorityQueue()
        # Add unique id to each entry so that we can compare two entries with
        # same counts.
        # Note that itreitems randomly order the entries.
        for uid, (w, c) in enumerate(six.iteritems(word_counts)):
            q.put((c, uid, w))

        while q.qsize() >= 2:
            (count1, id1, word1) = q.get()
            (count2, id2, word2) = q.get()
            count = count1 + count2
            tree = (word1, word2)
            q.put((count, min(id1, id2), tree))

        return q.get()[2]

    def __call__(self, x, t):
        """Computes the loss value for given input and ground truth labels.

        Args:
            x (~chainer.Variable): Input to the classifier at each node.
            t (~chainer.Variable): Batch of ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        f = copy.copy(self._func)  # creates a copy of the function node
        return f(x, t, self.W)

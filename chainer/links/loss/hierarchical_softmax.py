import copy

import numpy
import six

from chainer import cuda
from chainer import function
from chainer import link
from chainer.utils import type_check

tau_opt=1
if tau_opt:
    import tau_ext
    import _ctau

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
            o = self.forward_cpu_org(inputs)
            err = abs((n[0] - o[0]) / o[0])
            assert(err < 1.0e-4), (err, o, n)
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
        assert(t.shape == (M,)), (t.shape, x.shape)
        assert(W.shape == (P,N)), (W.shape, x.shape)
        assert(codes.shape == (pn,)), (codes.shape, paths.shape)
        assert(x.dtype == numpy.float32), x.dtype
        assert(t.dtype == numpy.int32), t.dtype
        assert(W.dtype == numpy.float32), W.dtype
        assert(begins.dtype == numpy.int32), begins.dtype
        assert(paths.dtype == numpy.int32), paths.dtype
        assert(codes.dtype == numpy.float32), codes.dtype
        max_length = _ctau.binary_hierarchical_softmax_function_max_length(t, M, begins)
        ls = numpy.zeros((M, length), dtype=numpy.float32)
        wxy = numpy.zeros_like(ls)
        f = _ctau.binary_hierarchical_softmax_function_forward_cpu
        l = f(x, t, W, begins, paths, codes, wxy, M, N, P, bn, pn, max_length)
        self.wxy = wxy
        return ls.sum(),

    def xxx_forward_cpu_new_xxx(self, inputs):
        x, t, W = inputs
        begins = self.begins
        paths = self.paths
        codes = self.codes
        M,N = x.shape
        P,_ = W.shape
        bn, = begins.shape
        pn, = paths.shape
        assert(t.shape == (M,)), (t.shape, x.shape)
        assert(W.shape == (P,N)), (W.shape, x.shape)
        assert(codes.shape == (pn,)), (codes.shape, paths.shape)
        assert(x.dtype == numpy.float32), x.dtype
        assert(t.dtype == numpy.int32), t.dtype
        assert(W.dtype == numpy.float32), W.dtype
        assert(begins.dtype == numpy.int32), begins.dtype
        assert(paths.dtype == numpy.int32), paths.dtype
        assert(codes.dtype == numpy.float32), codes.dtype
        # _ctaumodule.c:ctau_binary_hierarchical_softmax_function_forward_cpu_wrap
        f = _ctau.binary_hierarchical_softmax_function_forward_cpu
        l = f(x, t, W, begins, paths, codes, M, N, P, bn, pn)
        return numpy.array(l),
        
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
            if 0:
                gx_o,none_o,gW_o = self.backward_cpu_org(inputs, grad_outputs)
                x_err = ((gx_n - gx_o)**2).sum()
                assert(x_err <= 1.0e-10 * (gx_o**2).sum()), x_err
                W_err = ((gW_n - gW_o)**2).sum()
                assert(W_err <= 1.0e-10 * (gW_o**2).sum()), W_err
            return gx_n,none_n,gW_n
        else:
            return self.backward_cpu_org(inputs, grad_outputs)

    def xxx_backward_cpu_new_xxx(self, inputs, grad_outputs):
        x, t, W = inputs
        gloss, = grad_outputs
        begins = self.begins
        paths = self.paths
        codes = self.codes
        wxy = self.wxy
        gx = numpy.empty_like(x)
        gW = numpy.zeros_like(W)
        M,N = x.shape
        P,_ = W.shape
        bn, = begins.shape
        pn, = paths.shape
        _,max_length = wxy.shape
        assert(t.shape == (M,)), (t.shape, x.shape)
        assert(W.shape == (P,N)), (W.shape, x.shape)
        assert(codes.shape == (pn,)), (codes.shape, paths.shape)
        assert(x.dtype == numpy.float32), x.dtype
        assert(t.dtype == numpy.int32), t.dtype
        assert(W.dtype == numpy.float32), W.dtype
        assert(begins.dtype == numpy.int32), begins.dtype
        assert(paths.dtype == numpy.int32), paths.dtype
        assert(codes.dtype == numpy.float32), codes.dtype
        for i in range(M):
            it = t[i]           # scalar
            begin = self.begins[it] # scalar
            end = self.begins[it + 1] # scalar
            path = self.paths[begin:end] # 1D array (end-begin) elems
            w = W[path]                  # 2D array (end-begin, N)
            wxy = w.dot(x[i]) * self.codes[begin:end] # 1D array (end-begin) elems
            g = -gloss * self.codes[begin:end] / (1.0 + numpy.exp(wxy)) # 1D array (end-begin) elems
            gx_ = g.dot(w)      # 1D N elems
            gw = g.reshape((end - begin, 1)).dot(x[i].reshape(1, N)) # (end-begin, N)
            gW[path] += gw      # (end-begin, N)
            gx[i] = gx_         # 1D N
        return gx, None, gW

    def xxx_backward_cpu_new_xxx(self, inputs, grad_outputs):
        x, t, W = inputs
        gloss, = grad_outputs
        begins = self.begins
        paths = self.paths
        codes = self.codes
        gx = numpy.zeros_like(x)
        gW = numpy.zeros_like(W)
        M,N = x.shape
        P,_ = W.shape
        bn, = begins.shape
        pn, = paths.shape
        assert(t.shape == (M,)), (t.shape, x.shape)
        assert(W.shape == (P,N)), (W.shape, x.shape)
        assert(codes.shape == (pn,)), (codes.shape, paths.shape)
        assert(x.dtype == numpy.float32), x.dtype
        assert(t.dtype == numpy.int32), t.dtype
        assert(W.dtype == numpy.float32), W.dtype
        assert(begins.dtype == numpy.int32), begins.dtype
        assert(paths.dtype == numpy.int32), paths.dtype
        assert(codes.dtype == numpy.float32), codes.dtype
        for i in range(M):
            it = t[i]           # scalar
            begin = begins[it] # scalar
            end = begins[it + 1] # scalar
            for k in range(begin, end):
                p = paths[k] # 1D array 1 elems
                w = W[p]                  # 1D N
                wxy = w.dot(x[i]) * codes[k] # scalar
                g = -gloss * codes[k] / (1.0 + numpy.exp(wxy)) # scalar
                gW[p] += g * x[i] # 1D N
                gx[i] += g * w    # 1D N
        return gx, None, gW
    
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
            hist = {}
            for p in paths:
                hist[p] = hist.get(p, 0) + 1
            print("paths:")
            for p,c in sorted(hist.items()):
                print(p,c)
        
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
        # _ctaumodule.c:ctau_binary_hierarchical_softmax_function_backward_cpu_wrap
        f = _ctau.binary_hierarchical_softmax_function_backward_cpu
        f(x, t, W, gx, gW, begins, paths, codes, wxy, gloss, M, N, P, bn, pn, ml)
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

"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        # naive implementation 
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        stride = conv_param['stride']
        p = conv_param['pad']
        W_prime = (W + 2 * p -  WW) // stride + 1
        H_prime = (H + 2 * p - HH) // stride + 1
        out = torch.zeros(N, F, H_prime, W_prime, device=x.device, dtype=x.dtype)
        x_pad = torch.nn.functional.pad(x, (p, p, p, p), mode='constant', value=0)
        for i in range(N): 
            image = x_pad[i]
            for j in range(F):
                fil = w[j]
                for k in range(H_prime):
                    for l in range(W_prime):
                        spatial = image[:, (stride * k):(stride * k + HH) ,(stride * l):(stride * l + WW)]
                        out[i][j][k][l] = (spatial * fil).sum() + b[j]
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        # naive implementation 
        x, w, b, conv_param = cache
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        stride = conv_param['stride']
        p = conv_param['pad']
        W_prime = (W + 2 * p -  WW) // stride + 1
        H_prime = (H + 2 * p - HH) // stride + 1
        x_pad = torch.nn.functional.pad(x, (p, p, p, p), mode='constant', value=0)
        # dx = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        dx_pad = torch.zeros_like(x_pad, device=x.device, dtype=x.dtype)
        dw = torch.zeros_like(w, device=w.device, dtype=w.dtype)
        db = torch.zeros_like(b, device=b.device, dtype=b.dtype)
        for i in range(N):
            image = x_pad[i]
            for j in range(F):
                fil = w[j]
                for k in range(H_prime):
                    for l in range(W_prime):
                        spatial = image[:, (stride * k):(stride * k + HH) ,(stride * l):(stride * l + WW)]
                        dx_pad[i, :, (stride * k):(stride * k + HH), (stride * l):(stride * l + WW)] += fil * dout[i, j, k, l]
                        dw[j] += spatial * dout[i, j, k, l]
                        db[j] += dout[i, j, k, l]
        dx = dx_pad[:, :, p:(H + p), p:(W + p)]
        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        H_prime = (H - pool_height) // stride + 1
        W_prime = (W - pool_width) // stride + 1
        out = torch.zeros(N, C, H_prime, W_prime, device=x.device, dtype=x.dtype)
        for i in range(N):
            image = x[i]
            for j in range(C):
                for k in range(H_prime):
                    for l in range(W_prime):
                        spatial = image[j, (k * stride):(k * stride + pool_height),(l * stride):(l * stride + pool_width)]
                        out[i][j][k][l] = spatial.max()
                
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        x, pool_param = cache
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        H_prime = (H - pool_height) // stride + 1
        W_prime = (W - pool_width) // stride + 1
        dx = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        for i in range(N):
            image = x[i]
            for j in range(C):
                for k in range(H_prime):
                    for l in range(W_prime):
                        spatial = image[j, (k * stride):(k * stride + pool_height),(l * stride):(l * stride + pool_width)]
                        mask = spatial == spatial.max()
                        # 注意是 +=, x的一个元素可能在多个pool窗口中都是最大值
                        dx[i, j, (k * stride):(k * stride + pool_height),(l * stride):(l * stride + pool_width)] += mask * dout[i, j, k, l]
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        # 不用初始化conv_param和pool_param, 在loss中设置
        # same padding: p = (k - 1) / 2, stride = 1, H' = H, W' = W
        # 2 * 2 Max pool
        # filter_size说明filter是方阵 
        C, H, W = input_dims
        w1 = torch.zeros(num_filters, C, filter_size, filter_size,device=device,dtype=dtype)
        for i in range(num_filters):
            w1[i] = torch.randn(C, filter_size, filter_size, device=device, dtype=dtype) * weight_scale 
        self.params['W1'] = w1 
        self.params['b1'] = torch.zeros(num_filters, device=device, dtype=dtype) 
        # H = H', W = W'
        # 2 * 2 max pool subsample the output by factor 1/2
        # after max pool, (F, H//2, W//2)
        # linear的时候要将input flatten成vector
        self.params['W2'] = torch.randn(num_filters * (H // 2) * (W // 2), hidden_dim, device=device, dtype=dtype) * weight_scale
        self.params['b2'] = torch.zeros(hidden_dim, device= device, dtype=dtype)
        self.params['W3'] = torch.randn(hidden_dim, num_classes, device= device, dtype=dtype) * weight_scale
        self.params['b3'] = torch.zeros(num_classes, device= device, dtype=dtype)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        out, CRP_cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out, LR_cache = Linear_ReLU.forward(out, W2, b2)
        scores, L_cache = Linear.forward(out, W3, b3)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        reg = self.reg
        loss, dx = softmax_loss(scores, y) 
        # add L2 Regularization 
        loss += reg * (W1 ** 2).sum() + reg * (W2 ** 2).sum() + reg * (W3 ** 2).sum()
        dx, grads['W3'], grads['b3'] = Linear.backward(dx, L_cache)  
        grads['W3'] += 2 * reg * W3 
        dx, grads['W2'], grads['b2'] = Linear_ReLU.backward(dx, LR_cache)
        grads['W2'] += 2 * reg * W2 
        _ , grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dx, CRP_cache) 
        grads['W1'] += 2 * reg * W1 
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        C, H, W = input_dims
        filter_size = 3 
        pool_index = 0
        num_pools = len(max_pools)
        use_kaiming = False
        if weight_scale == 'kaiming':
            use_kaiming = True
        # L-1 macro layers
        # all conv layers
        # 1. same padding  
        # 2. kernel(filter)_size = 3, pad = 1
        # 3. max pool 2 * 2, stride = 2, assume max_pools is ascending
        for i in range(self.num_layers - 1):
            num_filter = num_filters[i]
            if use_kaiming:
                # weight_scale = sqrt(2 / Din)
                # for conv layer, Din = channels * kernel_size ** 2
                # note, the channel will update between different layers
                # weight_scale = torch.sqrt(torch.tensor([2 / (C * filter_size ** 2)], device=device, dtype=dtype))
                self.params[f'W{i+1}'] = kaiming_initializer(C, num_filter, filter_size, relu=True, device=device, dtype=dtype)
            else:
                w_temp = torch.zeros(num_filter, C, filter_size, filter_size, device=device, dtype=dtype)
                # initial every filter for i macro layer
                for j in range(num_filter):
                        # small randn or kaiming initialization
                        w_temp[j] = torch.randn(C, filter_size, filter_size, device=device, dtype=dtype) * weight_scale 
                self.params[f'W{i+1}'] = w_temp
            self.params[f'b{i+1}'] = torch.zeros(num_filter, device=device, dtype=dtype) 
            C = num_filter 
            if pool_index < num_pools and max_pools[pool_index] == i:
                pool_index += 1
                H //= 2
                W //= 2
            if batchnorm == True:
                # todo!, initialize the params with batch_norm 
                self.params[f'gamma{i+1}'] = torch.ones(num_filter, device=device, dtype=dtype)
                self.params[f'beta{i+1}'] = torch.zeros(num_filter, device=device, dtype=dtype)
            
            pass 
        # the last Linear Layer
        # the output of second last conv layer (num_filters[self.num_layers - 2], H, W) 
        if use_kaiming:
            # for linear layer
            # weight_scale = torch.sqrt(torch.tensor([2 / (num_filters[-1] * H * W)], device=device, dtype=dtype))
            self.params[f'W{self.num_layers}'] = kaiming_initializer(num_filters[-1] * H * W, num_classes, K=None, relu=False, device=device, dtype=dtype)
        else:
            self.params[f'W{self.num_layers}'] = torch.randn(num_filters[-1] * H * W, num_classes, device=device, dtype=dtype) * weight_scale
        self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, device=device, dtype=dtype)
            
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        caches = []
        pool_index = 0
        max_pools = self.max_pools
        num_pool = len(max_pools)
        x = X.clone()
        # first, without batchnorm
        for i in range(self.num_layers - 1):
            # L-1 macro layers
            if self.batchnorm == False:
                if pool_index < num_pool and max_pools[pool_index] == i:
                    pool_index += 1 
                    x, cache = Conv_ReLU_Pool.forward(x, self.params[f'W{i+1}'], self.params[f'b{i+1}'], conv_param, pool_param)
                else:
                    # without max_pool
                    x, cache = Conv_ReLU.forward(x, self.params[f'W{i+1}'], self.params[f'b{i+1}'], conv_param)
                caches.append(cache)
            else:
                # with batch normalization 
                if pool_index < num_pool and max_pools[pool_index] == i:
                    pool_index += 1
                    x, cache = Conv_BatchNorm_ReLU_Pool.forward(x, self.params[f'W{i+1}'], self.params[f'b{i+1}'], self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], conv_param, self.bn_params[i], pool_param)
                else:
                    # with bacth normalization, without max_pool
                    x, cache = Conv_BatchNorm_ReLU.forward(x, self.params[f'W{i+1}'], self.params[f'b{i+1}'], self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], conv_param, self.bn_params[i])
                caches.append(cache)
        # the last linear layer
        scores, cache = Linear.forward(x, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        caches.append(cache)
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        reg = self.reg
        num_layers = self.num_layers
        pool_index -= 1
        loss, dx = softmax_loss(scores, y) 
        # Linear Layers
        loss += reg * (self.params[f'W{self.num_layers}'] ** 2).sum()
        dx, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = Linear.backward(dx, caches.pop()) 
        grads[f'W{self.num_layers}'] += 2 * reg * self.params[f'W{self.num_layers}']
        # L-1 Conv layers
        for i in reversed(range(num_layers - 1)):
            # add L2 Regularization
            loss += reg * (self.params[f'W{i+1}'] ** 2).sum()

            if not self.batchnorm:
                if pool_index >= 0 and max_pools[pool_index] == i: 
                    pool_index -= 1
                    # with max pool
                    dx, grads[f'W{i+1}'], grads[f'b{i+1}'] = Conv_ReLU_Pool.backward(dx, caches.pop())
                else:
                    # without max pool
                    dx, grads[f'W{i+1}'], grads[f'b{i+1}'] = Conv_ReLU.backward(dx, caches.pop())
            else:
                # with normalization
                if pool_index >= 0 and max_pools[pool_index] == i: 
                    pool_index -= 1
                    # with max pool
                    dx, grads[f'W{i+1}'], grads[f'b{i+1}'], grads[f'gamma{i+1}'], grads[f'beta{i+1}'] = Conv_BatchNorm_ReLU_Pool.backward(dx, caches.pop())
                else:
                    # without max pool
                    dx, grads[f'W{i+1}'], grads[f'b{i+1}'], grads[f'gamma{i+1}'], grads[f'beta{i+1}'] = Conv_BatchNorm_ReLU.backward(dx, caches.pop())
                    
            # add L2 Regularization
            grads[f'W{i+1}'] += 2 * reg * self.params[f'W{i+1}']

        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale = 1e-1 
    learning_rate = 4e-3 
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    # todo!: add batch_nomalization
    input_dims = data_dict['X_train'][0].shape
    # VGG-Net style
    # num_filters = ([64] * 2) + ([128] * 2) + ([256] * 2) + ([512] * 8)
    num_filters = ([64] * 2) + ([128] * 2) + ([256] * 2) + ([512] * 1)   
    # max_pools = [1, 3, 5, 9]
    max_pools = [1, 3, 5, 6]
    weight_scale = 'kaiming'
    reg = 3e-5
    # CIFAR-10 especially
    model = DeepConvNet(input_dims, num_classes=10, num_filters=num_filters, max_pools=max_pools, batchnorm=False, weight_scale=weight_scale, reg=reg, dtype=dtype, device=device)
    # hyper params
    data = {
      'X_train': data_dict['X_train'],
      'y_train': data_dict['y_train'],
      'X_val': data_dict['X_val'],
      'y_val': data_dict['y_val'],
    }
    num_epochs = 3 
    batch_size = 128
    update_rule = adam
    optim_config = {'learing_rate' : 8e-2, }
    solver = Solver(model, data, num_epochs=num_epochs, batch_size=batch_size, update_rule=update_rule, optim_config=optim_config, device=device)
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        # linear
        weight = torch.randn(Din, Dout, device=device, dtype=dtype) * torch.sqrt(torch.tensor([gain / Din], device=device, dtype=dtype))
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        # conv layer
        # Din is input channels, Dout is output channels
        # weight = torch.randn(Din * K ** 2, Dout, device=device, dtype=dtype) * torch.sqrt(torch.tensor([gain / Din * K ** 2], device=device, dtype=dtype))
        # F(Dout) filters, each shape(C(Din), K, K)
        weight_scale = torch.sqrt(torch.tensor([gain / (Din * K ** 2)], device=device, dtype=dtype))
        weight = torch.zeros(Dout, Din, K, K, device=device, dtype=dtype)
        for i in range(Dout):
            weight[i] = torch.randn(Din, K, K, device=device, dtype=dtype) * weight_scale
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            '''
            # compute statistics 
            sample_mean = x.mean(dim=0)
            sample_var  = x.var(dim=0)
            # normalized data and rescale and shift
            x_hat = (x - sample_mean) / torch.sqrt(sample_var + eps)
            out = gamma * x_hat + beta
            # update running_mean and running_var
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            # cache
            cache = (x, x_hat, gamma, beta, sample_mean, sample_var, bn_param)
            '''
            # build from computation graph
            u = x.mean(dim=0)
            delta_mean = x - u
            delta_mean_sq = delta_mean ** 2
            var = delta_mean_sq.mean(dim=0)
            unit_var_recip = 1 / torch.sqrt(var + eps)
            x_hat = delta_mean * unit_var_recip
            out = gamma * x_hat + beta
            # update running_mean and running_var
            running_mean = momentum * running_mean + (1 - momentum) * u
            running_var = momentum * running_var + (1 - momentum) * var
            # cache 
            cache = (x, x_hat, gamma, delta_mean, unit_var_recip, var, bn_param)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            out = gamma * (x - running_mean) / torch.sqrt(running_var + eps) + beta
            # cache is None for test mode, no need to store anything for backward pass

            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code
        ''' naive implementation
        x, x_hat, gamma, delta_mean, unit_var_recip, var, bn_param = cache
        mode = bn_param['mode']
        # eps = bn_param['eps']
        eps = bn_param.get('eps', 1e-5)
        N = dout.shape[0]
        if mode == 'train':
            # build from computation graph
            # ddelta_mean来自于两个分支，一个是dout * gamma * unit_var_recip，另一个是ddelta_mean_sq * 2 * delta_mean
            dx = torch.zeros_like(dout)
            ddelta_mean = torch.zeros_like(dx)

            dbeta = dout.sum(dim=0)
            dgamma = (dout * x_hat).sum(dim=0)
            dx_hat = dout * gamma
            dunit_var_recip = (dx_hat * delta_mean).sum(dim=0)
            dvar = -0.5 * (var + eps) ** -1.5 * dunit_var_recip
            ddelta_mean_sq = torch.ones_like(x) / N * dvar

            # ddelta_mean += 2 * delta_mean * ddelta_mean_sq
            # ddelta_mean += 2 * delta_mean * dvar / N
            # ddelta_mean += dx_hat * unit_var_recip

            # ddelta_mean = dx_hat * unit_var_recip + (dx_hat * delta_mean).sum(dim=0) * -0.5 * (var + eps) ** -1.5 * 2 * delta_mean / N
            ddelta_mean = dx_hat * unit_var_recip + ((dx_hat * delta_mean) * -0.5 * (var + eps) ** -1.5 * 2).mean(dim=0) * delta_mean

            du = -ddelta_mean.sum(dim=0)
            dx += ddelta_mean
            dx += torch.ones_like(x) / N * du 
        else:
            # test mode, no need to compute the gradient
            pass
        ''' 
        # for faster backward pass in #Train a good model
        x, x_hat, gamma, delta_mean, unit_var_recip, var, bn_param = cache
        mode = bn_param['mode']
        # eps = bn_param['eps']
        eps = bn_param.get('eps', 1e-5)
        N = dout.shape[0]
        dbeta = dout.sum(dim=0)
        dgamma = (dout * x_hat).sum(dim=0)
        # dx = unit_var_recip * (dy - torch.mean(dy, dim=0) - x_hat * torch.mean(dy * x_hat, dim=0))
        dx = (1. / N) * gamma * unit_var_recip * (N * dout - dout.sum(dim=0) - x_hat * (dout * x_hat).sum(dim=0))
        dy = dout * gamma
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Replace "pass" statement with your code
        x, x_hat, gamma, delta_mean, unit_var_recip, var, bn_param = cache
        mode = bn_param['mode']
        # eps = bn_param['eps']
        eps = bn_param.get('eps', 1e-5)
        N = dout.shape[0]
        dbeta = dout.sum(dim=0)
        dgamma = (dout * x_hat).sum(dim=0)
        # dx = unit_var_recip * (dy - torch.mean(dy, dim=0) - x_hat * torch.mean(dy * x_hat, dim=0))
        dx = (1. / N) * gamma * unit_var_recip * (N * dout - dout.sum(dim=0) - x_hat * (dout * x_hat).sum(dim=0))
        dy = dout * gamma
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        out, cache = BatchNorm.forward(x, gamma, beta, bn_param)
        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        N, C, H, W = dout.shape
        dout = dout.permute(0, 2, 3, 1).reshape(-1, C)
        dx, dgamma, dbeta = BatchNorm.backward(dout, cache)
        dx = dx.reshape(N, H, W, C).permute(0, 3, 1, 2)
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta

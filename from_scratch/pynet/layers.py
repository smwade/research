"""
Layers for neural network.

Each layer consits of two methods, the forward and the backward pass.
The forward simply computes the output and caches inputs.
The back propegation takes the upstream derivative and the cache to 
produce the derivate with respect to all the inputs.
"""

import numpy as np

def linear_forward(x, w, b):


    """ Forward pass for fully-connected layer.
    Args:
      x : (np.array) input datai, shape [batch_size, d_1, ..., d_k]
      w : (np.array) weight tensor, shape [D, M]
      b : (np.array) bias, shape [M,]

    Returns:
      out : output of layer, shape [batch_size, M]
      cache: cache of inputs for backward pass (x, w, b)
    """
    BS = x.shape[0]
    flatten_shape = np.prod(x.shape[1:])
    x_flat = x.reshape(BS, flatten_shape)
    out = np.dot(x_flat, w) + b

    cache = (x, w, b)
    return out, cache

def linear_backward(dout, cache):
    """ Backward pass of fully-connected layer.
    Args:
      dout : (np.array) upstream derivative, shape [BS, M]
      cache: (tuple)
        - x : input data
        - w : weights
        - b : bias

    Returns:
      dx : (np.array) gradient with respect to x, shape [BS, d_1, .., d_k]
      dw : (np.array) gradient with respect to w, shape [D,M]
      db : (np.array) gradient with respect to b, shape [M,]
    """
    x, w, b = cache

    BS = x.shape[0]
    flatten_shape = np.prod(x.shape[1:])
    x_flat = x.reshape(BS, flatten_shape)

    dw = np.dot(x_flat.T, dout)
    dx = np.dot(dout, w.T).reshape(x.shape)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def relu_forward(x):
    """ Forward pass of relu activation.
    Args:
      x : (np.array) inputs

    Returns:
      out : (np.array) out, same shape as x
      cache : x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """ Backward pass of relu activation.
    Args:
      dout : upstream derivative
      cache : input x
    
    Returns:
      dx : gradiwnd with respect to x
    """
    x = cache
    out = np.maximum(0, x)
    out[out > 0] = 1
    dx = out * dout
    return dx


def softmax_loss(x, y):
    """ Loss and gradient for softmax classification.
    Args:
      x : input data, shape [BS, C]
      y : vector of labels [BS,]

    Returns:
      loss : scalar of the loss
      dx : gradient of loss with respect to x
    """
    y = y.astype(int)
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    BS = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(BS), y])) / BS
    dx = probs.copy()
    dx[np.arange(BS), y] -= 1
    dx /= BS
    return probs, loss, dx

def im2col(x, HH, WW, pad=0, stride=1):
    """ A helper function for faster convolutions. Transforms input so the 
    convolution can be performed as matrix multiplication.
    Args:
      x : Input array
      HH : kernal hight
      WW : kernal width
      pad : padding
      stride : the stride

    Returns:
      cols : (K**2 * c by B matrix)
    """
    B, D, H, W = x.shape
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    i0 = np.repeat(np.arange(HH), WW)
    i0 = np.tile(i0, D)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(WW), HH * D)
    j1 = stride * np.tile(np.arange(W_out), H_out)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(D), HH * WW).reshape(-1, 1)

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(HH * WW * D, -1)
    return cols


def col2im(cols, x_shape, HH=3, WW=3, pad=1,
                   stride=1):
    """ A helper function for faster convolutions.  This is the inverse of im2col,
    takes a col and converts it back to its propper shape.
    Args:
      cols : the cols to convert
      x_shape : shape to convert to
      HH : kernal hight
      WW : kernal width
      pad : padding
      stride : stride
    
    Returns:
      im : convoluted image
    """
    B, D, H, W = x_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    x_padded = np.zeros((B, D, H_padded, W_padded), dtype=cols.dtype)
    
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    i0 = np.repeat(np.arange(HH), WW)
    i0 = np.tile(i0, D)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(WW), HH * D)
    j1 = stride * np.tile(np.arange(W_out), H_out)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(D), HH * WW).reshape(-1, 1)
    
    cols_reshaped = cols.reshape(D * HH * WW, -1, B)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]

def conv_forward(x, k, b, pad, stride):
    """ The forward convolution implemented by matrix multiplication.
    Args:
      x : input data
      k : convolutional kernal
      b : the bias weights
      pad : padding
      stride : stride

    Returns:
      out : the activation of convolution
      cache : cached used parameters
    """
    B, D, H, W = x.shape
    F, D, HH, WW = k.shape
   
    # Check dimensions
    assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    assert (H + 2 * pad - HH) % stride == 0, 'height does not work'
    
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    out = np.zeros((B, F, H_out, W_out), dtype=x.dtype)
    
    x_cols = im2col(x, HH, WW, pad, stride)
    # flatten kernal
    result = k.reshape((F, -1)).dot(x_cols) #+ b.reshape(-1, 1)
    out = result.reshape(F, H_out, W_out, B)
    out = out.transpose(3, 0, 1, 2)
    
    cache = (x, k, b, pad, stride, x_cols)
    return out, cache

def conv_backward(dout, cache):
    """ The backward convolution implemented by matrix multiplication.
    Args:
      dout : the gradient of following layer
      cache : the cached variables from forward pass

    Returns:
      dx : dirivative with respect to x
      dw : dirivative with respect to w
      db : dirivative with respect to b
    """
    x, k, b, pad, stride, x_cols = cache
    
    db = np.sum(dout, axis=(0, 2, 3))

    F, D, HH, WW = k.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(k.shape)

    dx_cols = k.reshape(F, -1).T.dot(dout_reshaped)
    dx = col2im(dx_cols, x.shape, HH, WW, pad, stride)

    return dx, dw, db

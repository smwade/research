import numpy as np

"""
Implements various first-order update rules that are commonly used for
training neural networks.
"""

def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  next_w = w - config['learning_rate'] * dw
  return next_w, config


def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    (Setting momentum = 0 reduces to sgd.)
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  
  v = config['momentum']*v - config['learning_rate']*dw
  config['velocity'] = v

  next_w = w + v
  return next_w, config



def rmsprop(w, dw, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(x))

  config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw**2
  next_w = w - config['learning_rate'] * dw / (np.sqrt(config['cache']) + config['epsilon'])

  return next_w, config


def adam(w, dw, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(w))
  config.setdefault('v', np.zeros_like(w))
  config.setdefault('t', 0)
  
  config['t'] += 1
  config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dw
  config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dw**2)

  mt = config['m'] / (1-config['beta1']**config['t'])
  vt = config['v'] / (1-config['beta2']**config['t'])

  next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])
  
  return next_w, config

  
  
  


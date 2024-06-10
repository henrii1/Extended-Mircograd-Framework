import random
from collections.abc import Iterable

def zero_grad (func):
  def reset_gradients(obj):
    if hasattr(obj, 'parameters') and callable(getattr(obj, 'parameters')):
      for p in obj.parameters():
        p.grad = 0.0
  def wrapper(*args, **kwargs):
    for arg in args:
      if isinstance(arg, Iterable) and not isinstance(arg, (str, bytes)):
        for item in arg:
          reset_gradients(item)
      else:
        reset_gradients(arg)
    for kwarg in kwargs.values():
      if isinstance(kwarg, Iterable) and not isinstance(kwarg, (str, bytes)):
        for item in kwarg:
          reset_gradients(item)
        else:
          reset_gradients(kwarg)
    return func(*args, **kwargs)
  return wrapper
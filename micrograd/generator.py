import random
import math

class TValue:
    def __init__(self, data, _children=(), _operation="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _operation
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, TValue) else TValue(other)
        out = TValue(self.data + other.data, (self, other), "+")
        
        def _backward():  # '+' is a pass-through operation
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, TValue) else TValue(other)
        out = TValue(self.data * other.data, (self, other), "*")

        def _backward():  # '*' local derivative is the other data multiplied by out.grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = TValue(self.data **other, (self,), "**")

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1)  # we have already defined the '*' operator, that's why.
  
    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**(-1)
  
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
  
    def __rtruediv__(self, other):
        return other * self**-1
  
    def __rsub__(self, other):
        return other + (-self)

    def tanh(self):
        t = (math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1)
        out = TValue(t, (self, ), 'tanh')
    

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = TValue(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
    
      topo = []
      visited = set()
      def build_topo(v):
        if v not in visited:
          visited.add(v)
          for child in v._prev:
            build_topo(child)
          topo.append(v)
      build_topo(self)
      
      self.grad = 1.0
      for node in reversed(topo):
        node._backward()

    def __repr__(self):
        return f'TValue(data={self.data}, grad={self.grad})'
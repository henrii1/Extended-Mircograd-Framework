from micrograd.generator import TValue

# inputs x1,x2
x1 = TValue(2.0, label='x1')
x2 = TValue(0.0, label='x2')
# weights w1,w2
w1 = TValue(-3.0, label='w1')
w2 = TValue(1.0, label='w2')
# bias of the neuron
b = TValue(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()
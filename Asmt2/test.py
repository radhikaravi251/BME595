from logic_gates import AND
from logic_gates import NOT
from logic_gates import OR
from logic_gates import XOR
from neural_network import NeuralNetwork
import torch

# Logic Gates
And = AND()
Not = NOT()
Or = OR()
Xor = XOR()

# AND
print(And(False, False))
print(And(False, True))
print(And(True, False))
print(And(True, True))

# NOT
print(Not(False))
print(Not(True))

# OR
print(Or(False, False))
print(Or(False, True))
print(Or(True, False))
print(Or(True, True))

# XOR
print(Xor(False, False))
print(Xor(False, True))
print(Xor(True, False))
print(Xor(True, True))


# Test Case: Matt Mazur
test_nn = NeuralNetwork(2, 2, 2)
theta = test_nn.getLayer(0)
theta[0,0] = 0.35
theta[1,0] = 0.15
theta[2,0] = 0.20
theta[0,1] = 0.35
theta[1,1] = 0.25
theta[2,1] = 0.30
#print(theta)

theta = test_nn.getLayer(1)
theta[0,0] = 0.60
theta[1,0] = 0.40
theta[2,0] = 0.45
theta[0,1] = 0.60
theta[1,1] = 0.50
theta[2,1] = 0.55
#print(theta)

input_vector = torch.DoubleTensor([[0.05], [0.10]])
print(input_vector)
result = test_nn.forward(input_vector)
print (result)
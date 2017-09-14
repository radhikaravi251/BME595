from logicGates import AND
from logicGates import NOT
from logicGates import OR
from logicGates import XOR
from NeuralNetwork import NeuralNetwork
import torch


## Test Case: Matt Mazur
#test_nn = NeuralNetwork(2, 2, 2)
#theta = test_nn.getLayer(0)
#theta[0,0] = 0.35
#theta[1,0] = 0.15
#theta[2,0] = 0.20
#theta[0,1] = 0.35
#theta[1,1] = 0.25
#theta[2,1] = 0.30
##print(theta)
#
#theta = test_nn.getLayer(1)
#theta[0,0] = 0.60
#theta[1,0] = 0.40
#theta[2,0] = 0.45
#theta[0,1] = 0.60
#theta[1,1] = 0.50
#theta[2,1] = 0.55
##print(theta)
#
#input_vector = torch.DoubleTensor([[0.05], [0.10]])
#target_vector = torch.DoubleTensor([[0.01], [0.99]])
#print(input_vector)
#result = test_nn.forward(input_vector)
#print (result)
#
#for i in range(10000):
#    test_nn.forward(input_vector)
#    test_nn.backward(target_vector)
#    test_nn.updateParams(0.50)
#
#result = test_nn.forward(input_vector)
#print (result)

# Logic Gates
And = AND()
Not = NOT()
Or = OR()
Xor = XOR()

# AND
print("And(False, False): ", And(False, False))
print("And(False, True): ", And(False, True))
print("And(True, False): ", And(True, False))
print("And(True, True): ", And(True, True))

# NOT
print("Not(False): ", Not(False))
print("Not(True): ", Not(True))

# OR
print("Or(False, False): ", Or(False, False))
print("Or(False, True): ", Or(False, True))
print("Or(True, False): ", Or(True, False))
print("Or(True, True): ", Or(True, True))

# XOR
print("Xor(False, False): ", Xor(False, False))
print("Xor(False, True): ", Xor(False, True))
print("Xor(True, False): ", Xor(True, False))
print("Xor(True, True): ", Xor(True, True))


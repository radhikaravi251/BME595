from neural_network import NeuralNetwork
import numpy as np
import torch

class AND:
    def __init__(self):
        self.__nn__ = NeuralNetwork(2, 1)
        theta = self.__nn__.getLayer(0)
        theta[0,0] = -40.0
        theta[1,0] = 30.0
        theta[2,0] = 30.0
        
    def __call__(self, x, y):
        self.__inputVector = torch.DoubleTensor([[int(x)], [int(y)]])
        return self.forward()
        
    def forward(self):
        result = self.__nn__.forward(self.__inputVector)
        result = torch.round(result)
        return bool(result[0,0])

class OR:
    def __init__(self):
        self.__nn__ = NeuralNetwork(2, 1)
        theta = self.__nn__.getLayer(0)
        theta[0,0] = -20.0
        theta[1,0] = 30.0
        theta[2,0] = 30.0

    def __call__(self, x, y):
        self.__inputVector = torch.DoubleTensor([[int(x)], [int(y)]])
        return self.forward()
        
    def forward(self):
        result = self.__nn__.forward(self.__inputVector)        
        result = torch.round(result)
        return bool(result[0,0])
        
class NOT:
    def __init__(self):
        self.__nn__ = NeuralNetwork(1, 1)
        theta = self.__nn__.getLayer(0)
        theta[0,0] = 20.0
        theta[1,0] = -40.0

    def __call__(self, x):
        self.__inputVector = torch.DoubleTensor([[int(x)]])
        return self.forward()
        
    def forward(self):
        result = self.__nn__.forward(self.__inputVector)
        result = torch.round(result)
        return bool(result[0,0])

        
class XOR:
    def __init__(self):
        self.__nn__ = NeuralNetwork(2, 2, 1)
        theta = self.__nn__.getLayer(0)
        theta[0,0] = -40.0
        theta[1,0] = 70.0
        theta[2,0] = -40.0
        theta[0,1] = -40.0
        theta[1,1] = -40.0
        theta[2,1] = 70.0
        
        theta = self.__nn__.getLayer(1)
        theta[0,0] = -20.0
        theta[1,0] = 30.0
        theta[2,0] = 30.0

    def __call__(self, x, y):
        self.__inputVector = torch.DoubleTensor([[int(x)], [int(y)]])
        return self.forward()
        
    def forward(self):
        result = self.__nn__.forward(self.__inputVector)
        result = torch.round(result)
        return bool(result[0,0])
        
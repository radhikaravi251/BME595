import numpy as np
import torch

class NeuralNetwork:
    def __init__(self, *args):
        self.__in_layer = args[0]
        self.__out_layer = args[len(args) - 1]
        self.__n_layer = len(args)
        self.__theta = []
        for i in range(len(args) - 1):
            temp = np.random.normal(0, 1.0/((args[i])**(1.0/2)), ((args[i]+1),(args[i+1])))
            self.__theta.append(torch.DoubleTensor(temp))
    
    def __sigmoidF(self, input_vector):
        rows = input_vector.size(0)
        cols = input_vector.size(1)
        for i in range(rows):
            for j in range(cols):
                input_vector[i,j] = 1.0/(1.0 + np.exp(-input_vector[i,j]))
        return input_vector
        
    
    def getLayer(self, layer):
        return self.__theta[layer]
    
    def forward(self, input_vector):
        rows = input_vector.size(0)
        cols = input_vector.size(1)
        temp = torch.ones(1,cols)
        temp = temp.double()
        M = input_vector
        N = torch.DoubleTensor()
        for i in range(self.__n_layer - 1):
            M = torch.cat([temp, M], 0)
            N = torch.mm(torch.t(self.getLayer(i)), M)
            M = self.__sigmoidF(N)
        return M


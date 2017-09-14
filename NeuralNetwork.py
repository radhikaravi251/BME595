import numpy as np
import torch

class NeuralNetwork:
    def __init__(self, *args):
        self.build(*args)
            
    def build(self, *args):
        self.__in_layer = args[0]
        self.__out_layer = args[len(args) - 1]
        self.__n_layer = len(args)
        self.Theta = {}
        self.dE_dTheta = {}
        for i in range(len(args) - 1):
            temp = np.random.normal(0, 1.0/((args[i])**(1.0/2)), ((args[i]+1),(args[i+1])))
            self.Theta[i] = (torch.DoubleTensor(temp))
            
    
    def __sigmoidF(self, input_vector):
        rows = input_vector.size(0)
        cols = input_vector.size(1)
        for i in range(rows):
            for j in range(cols):
                input_vector[i,j] = 1.0/(1.0 + np.exp(-input_vector[i,j]))
        return input_vector
    
    def __sigmoidDerivative(self, output):
        return output*(1.0 - output)
    
    def getLayer(self, layer):
        return self.Theta[layer]
    
    def forward(self, input_vector):
        rows = input_vector.size(0)
        cols = input_vector.size(1)
        temp = torch.ones(1,cols)
        temp = temp.double()
        M = input_vector
        N = torch.DoubleTensor()
        self.__output = {}
        self.__outputBeforeActivation = {}
        self.__output[0] = M
        self.__outputBeforeActivation[0] = M        
        for i in range(self.__n_layer - 1):
            M = torch.cat([temp, M], 0)
            N = torch.mm(torch.t(self.getLayer(i)), M)
            self.__outputBeforeActivation[i+1] = N
            M = self.__sigmoidF(N)
            self.__output[i+1] = M
        return M

    def backward(self, target):
        rows = target.size(0)
        cols = target.size(1)
        self.deltaTest = {}
        temp = torch.zeros(1,cols)
        temp = temp.double()
        
        for i in range(self.__n_layer - 1, -1, -1):
            
            if i != self.__n_layer - 1:
                self.deltaTest[i] = torch.DoubleTensor(self.Theta[i].size(0)-1,1).zero_()
                self.deltaTest[i] = torch.DoubleTensor(self.Theta[i].size(0)-1,1).zero_()
                self.dE_dTheta[i] = torch.DoubleTensor(self.Theta[i].size(0),self.Theta[i].size(1)).zero_()
                for neuron in range(self.Theta[i].size(0)-1):
                    error = 0.0
                    for j in range(self.deltaTest[i+1].size(0)):
                        error = error + (self.deltaTest[i+1][j]*self.Theta[i][neuron+1][j])
                    self.deltaTest[i][neuron] = (error*self.__sigmoidDerivative(self.__outputBeforeActivation[i][neuron]))
            else:
                self.deltaTest[i] = (self.__output[i] - target)*self.__sigmoidDerivative(self.__outputBeforeActivation[i])
#                self.deltaTest[i] = torch.cat([temp, (self.__output[i] - target)*self.__sigmoidDerivative(self.__outputBeforeActivation[i])],0)
            
            if i != self.__n_layer - 1:
                for n1 in range(self.Theta[i].size(0)):
                    for n2 in range(self.Theta[i].size(1)):
                        if n1 == 0:
                            self.dE_dTheta[i][n1][n2] = torch.sum(self.deltaTest[i+1][n2])/float(self.deltaTest[i+1][n2].size(0))
                        else:
                            self.dE_dTheta[i][n1][n2] = 0.0
                            for nCols in range(self.deltaTest[i+1][n2].size(0)):
                                self.dE_dTheta[i][n1][n2] = self.dE_dTheta[i][n1][n2] + (self.deltaTest[i+1][n2][nCols]*self.__output[i][n1-1][nCols])
                            self.dE_dTheta[i][n1][n2] = self.dE_dTheta[i][n1][n2]/float(self.deltaTest[i+1][n2].size(0))
    
    def updateParams(self, eta):
        for i in range(self.__n_layer-1):
            for j in range(self.Theta[i].size(0)):
                for k in range(self.Theta[i].size(1)):
                    self.Theta[i][j][k] = self.Theta[i][j][k] - (eta*self.dE_dTheta[i][j][k])
#        print (self.Theta)
        
            

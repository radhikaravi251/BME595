import cv2
import numpy as np

class Conv2D:
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
		
    def forward(self, input_image):
        k1 = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
        k2 = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
        k3 = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
        k4 = np.array([[-1, -1, -1, -1, -1],[-1, -1, -1, -1, -1],[0, 0, 0, 0, 0],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
        k5 = np.array([[-1, -1, 0, 1, 1],[-1, -1, 0, 1, 1],[-1, -1, 0, 1, 1],[-1, -1, 0, 1, 1],[-1, -1, 0, 1, 1]])
        kernel = np.zeros(shape = (self.kernel_size, self.kernel_size, self.o_channel))
        if self.mode == 'known' and self.o_channel == 1 and self.kernel_size == 3:
            kernel[:,:,0] = k1
        
        elif self.mode == 'known' and self.o_channel == 2 and self.kernel_size == 5:
            kernel[:,:,0] = k4
            kernel[:,:,1] = k5
        
        elif self.mode == 'known' and self.o_channel == 3 and self.kernel_size == 3:
            kernel[:,:,0] = k1
            kernel[:,:,1] = k2
            kernel[:,:,2] = k3
            
        elif self.mode == 'rand':
            for nk in range(self.o_channel):
                kernel[:,:,nk] = np.random.random((self.kernel_size,self.kernel_size))
            
        x, y, z = input_image.shape
        x_new = int(np.floor(((x - self.kernel_size)/self.stride) + 1))
        y_new = int(np.floor(((y - self.kernel_size)/self.stride) + 1))
        z_new = int(self.o_channel)
        
        num_ops = 0
        
        output_image = np.zeros(shape = (x_new, y_new, z_new))
        
        for ch in range(0, z_new):
            for i in range(0, x_new):
                for j in range(0, y_new):
                    kernel_used = np.dstack([kernel[:,:,ch]]*self.in_channel)
                    block = (input_image[(self.stride*i):(self.stride*i)+self.kernel_size, (self.stride*j):(self.stride*j)+self.kernel_size,:]*kernel_used)
                    output_image[i][j][ch] = np.sum(np.sum(np.sum(block, axis = 0), axis = 0),axis=0)
                    num_ops = num_ops + (self.kernel_size*self.kernel_size*self.in_channel) + (self.in_channel*(self.kernel_size*self.kernel_size - 1)) + self.in_channel - 1
                    
        return num_ops, output_image
		
				
# HW 2: Radhika Ravi 0029200602

## Feedforward Pass
In order to set up the neural network, the size (excluding the bias node) of each layer is passed as an argument. Accordingly, a neural network is initialized with a bias node in the input layer as well as each hidden layer, with randomly generated values assigned to the weights.  
This function was tested with the example discussed at the following link: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/. The results match exactly.  

## Logic Gates
### AND:
This logic gate is implemented using a neural network with the following characteristics:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/AND.jpg)

### OR:
This logic gate is implemented using a neural network with the following characteristics:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/OR.jpg)

### NOT:
This logic gate is implemented using a neural network with the following characteristics:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/NOT.jpg)

### XOR:
This logic gate is implemented as a breakdown of XOR as: (*x* XOR *y*) = (*x* AND *~y*) OR (*~x* AND *y*). So, the hidden layer nodes denote: *h1* = (*x* AND *~y*) and *h2* = (*~x* AND *y*). The neural network has the following characteristics:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/XOR.jpg)

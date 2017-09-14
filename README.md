# HW 3: Radhika Ravi 0029200602

## Backpropagation
In order to run the python script:  
import torch
from NeuralNetwork import NeuralNetwork
from logicGates import AND
from logicGates import OR
from logicGates import NOT
from logicGates import XOR
img2num.py

The backpropagation works correctly according to the example given in: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

On training with 10,000 iterations, the resultant output values are obtained to be 0.0119 (against 0.01) and 0.9881 (against 0.99).

## Logic Gates
The learning rate was set to be 0.20. The number of training samples used were 1000 and a total of 100 iterations were performed for each sample to update the weights.  
The weights obtained previously are shown in:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/AND.jpg)  
The corresponding newly obtained weights are:  
Old: -40, 30, 30  
New: -10.8277, 7.1634, 7.1152

### OR
The weights obtained previously are shown in:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/OR.jpg)  
The corresponding newly obtained weights are:  
Old: -20, 30, 30  
New: -3.7128, 7.9054, 7.9243

### NOT
The weights obtained previously are shown in:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/NOT.jpg)  
The corresponding newly obtained weights are:  
Old: 20, -40  
New: 4.3617, -8.9070 

### XOR
The weights obtained previously are shown in:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/XOR.jpg)  
The corresponding newly obtained weights are:  
Old: Input to Hidden: (-40, 70, -40), (-40, -40, 70);  Hidden to Output: (-20, 30, 30)  
New: Input to Hidden: (-4.1, 0.9148, 0.4976), (-0.5678, 0.1580);  Hidden to Output: (-20, 30, 30)  

It can be clearly observed that the relative values of the weights as set manually and as obtained by training the neural network show the same trend, thus leading to the correct results. 

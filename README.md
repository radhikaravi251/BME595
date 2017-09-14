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

On training with 10,000 iterations, the resultant output values are obtained to be 0.0116 (against 0.01) and 0.9885 (against 0.99).

## Logic Gates
The learning rate was set to be 0.20. The number of training samples used were 1000 and a total of 100 iterations were performed for each sample to update the weights.  
The weights obtained previously are shown in:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/AND.jpg)  
The corresponding newly obtained weights are:  
Old: -40, 30, 30  
New: -10.8490, 7.2810, 7.1819

### OR
The weights obtained previously are shown in:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/OR.jpg)  
The corresponding newly obtained weights are:  
Old: -20, 30, 30  
New: -3.7025, 7.8999, 7.9405

### NOT
The weights obtained previously are shown in:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/NOT.jpg)  
The corresponding newly obtained weights are:  
Old: 20, -40  
New: 4.3194, -8.9307 

### XOR
The weights obtained previously are shown in:  
![](https://github.com/radhikaravi251/BME595/blob/Asmt2/Asmt2/XOR.jpg)  
The corresponding newly obtained weights are:  
Old: Input to Hidden: (-40, 70, -40), (-40, -40, 70);  Hidden to Output: (-20, 30, 30)  
New: Input to Hidden: (2.8041, 5.7339, -5.5726), (-3.5199, 6.4355, -6.4993);  Hidden to Output: (4.4238, -9.2802, 9.3402)   

It can be clearly observed that the relative values of the weights as set manually and as obtained by training the neural network show the same trend, thus leading to the correct results. 

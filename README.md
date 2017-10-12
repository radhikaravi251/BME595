# Title  
CNN-based Lane Detection from Highway Imagery Dataset  

## Team members  
Yun-Jou Lin(Roselin), Radhika Ravi(radhikaravi251)  

## Motivation
Lane detection is one of critical elements in autonomous driving. 
Currently, there are many sensors used in autonomous driving such as laser scanner, radar, and camera to detect objects.
However, laser scanners and radars are expensive. 
Therefore, cameras could be a cost-down option for autonomous driving system.

## Goals  
Identify the lane marking on the highway imagery.  
![Image1](https://github.com/Roselin/DeepLearning/blob/master/Project/flea2_77_front_2017-05-02-134716-0242.jpg)

## Challenges  
The illumination condition on image could be different from place, weather, and time.
Therefore, a huge training dataset should be collected to create a robust neural network.
The target(labelling) of training dataset is difficult to create.  
Moreover, manual labelling could cause the human errors. 
Since the lane marking can provide high retro-reflective to laser scanner, we can collect the lane marking point cloud from laser scanners and project to imagery to get the training dataset.

 

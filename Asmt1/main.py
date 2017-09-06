from conv import Conv2D
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

## PART A
image1 = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

cv2.imwrite('image1_original.jpg', image1)
cv2.imwrite('image2_original.jpg', image2)

conv2d_task1 = Conv2D(3, 1, 3, 1, 'known')
conv2d_task2 = Conv2D(3, 2, 5, 1, 'known')
conv2d_task3 = Conv2D(3, 3, 3, 2, 'known')

# Part A: Task 1
num_operations1, filtered_image1 = conv2d_task1.forward(image1)
num_operations2, filtered_image2 = conv2d_task1.forward(image2)

cv2.imwrite('image1_PartA_Task1.jpg', filtered_image1)
cv2.imwrite('image2_PartA_Task1.jpg', filtered_image2)

# Part A: Task 2
num_operations1, filtered_image1 = conv2d_task2.forward(image1)
num_operations2, filtered_image2 = conv2d_task2.forward(image2)

cv2.imwrite('image1_PartA_Task2_channel1.jpg', filtered_image1[:,:,0])
cv2.imwrite('image1_PartA_Task2_channel2.jpg', filtered_image1[:,:,1])

cv2.imwrite('image2_PartA_Task2_channel1.jpg', filtered_image2[:,:,0])
cv2.imwrite('image2_PartA_Task2_channel2.jpg', filtered_image2[:,:,1])

# Part A: Task 3
num_operations1, filtered_image1 = conv2d_task3.forward(image1)
num_operations2, filtered_image2 = conv2d_task3.forward(image2)

cv2.imwrite('image1_PartA_Task3_channel1.jpg', filtered_image1[:,:,0])
cv2.imwrite('image1_PartA_Task3_channel2.jpg', filtered_image1[:,:,1])
cv2.imwrite('image1_PartA_Task3_channel3.jpg', filtered_image1[:,:,2])

cv2.imwrite('image2_PartA_Task3_channel1.jpg', filtered_image2[:,:,0])
cv2.imwrite('image2_PartA_Task3_channel2.jpg', filtered_image2[:,:,1])
cv2.imwrite('image2_PartA_Task3_channel3.jpg', filtered_image2[:,:,2])


## PART B
time_taken = []
x = []
for num_ch in range(11):
    o_channel = 2**num_ch
    x.append(num_ch)
    conv2d_partB = Conv2D(3, o_channel, 3, 1, 'rand')
    start = time.clock()
    num_operations, filtered_image = conv2d_partB.forward(image1)
    time_taken.append(time.clock() - start)
    
plt.plot(x, time_taken)
plt.title('Plot of time taken')
plt.xlabel('Number of output channels')
plt.ylabel('Time taken')
plt.savefig('Time_Plot.jpg')
plt.close()

## PART C
number_of_operations = []
x=[]
for i in range(5):
    kernel_size = 2*(i+1) + 1
    x.append(kernel_size)
    conv2d_partC = Conv2D(3, 2, kernel_size, 1, 'rand')
    num_operations, filtered_image = conv2d_partC.forward(image1)
    number_of_operations.append(num_operations)
    
plt.plot(x, number_of_operations)
plt.title('Plot of number of operations')
plt.xlabel('Kernel size')
plt.ylabel('Number of operations')
plt.savefig('Operations_Plot.jpg')
plt.close()

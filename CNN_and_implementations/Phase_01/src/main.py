import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST
from convolution import convolve2d
from pooling import max_pool2d, average_pool2d

# Load dataset

dataset= FashionMNIST(
    root="datasets",
    train= True,
    download= False
)

# Get first image
image, label= dataset[0]

#Convert PIL image to numpy
image= np.array(image)

#Simple edge detection kernel
kernel = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

# Running convolution
output= convolve2d(
    image= image,
    kernel= kernel,
    stride=1,
    padding=1
)

# Applying pooling
max_pooled= max_pool2d(np.abs(output))
avg_pooled= average_pool2d(np.abs(output))

#Show original +result
plt.figure(figsize= (16,4))

plt.subplot(1,4,1)
plt.imshow(image, cmap= 'gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(np.abs(output),cmap='gray') 

"""
- By doing np.abs(output) negative edges were visible too
- with using only "output" only positive edges were visible

- what do you mean by negative edges?
"""
plt.title("Convolved")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(max_pooled,cmap='gray')
plt.title("Max Pooling")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(avg_pooled,cmap='gray')
plt.title("average Pooling")
plt.axis("off")

plt.show()
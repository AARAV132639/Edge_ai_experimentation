import numpy as np
from torchvision.datasets import CIFAR10

import sys
import os
# This tells python to look one folder up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

#Phase 2 imports
from Phase_02_rgb.src.multi_filter_convolution import multi_filter_convolve
from Phase_02_rgb.src.activations import relu
from Phase_02_rgb.src.pooling import max_pool_multi_channel

# Phase 3 imports
from src.flatten import flatten
from src.dense import DenseLayer
from src.softmax import softmax

# CIFAR-10 class names
classes= [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Loading datasets

current_dir= os.path.dirname(os.path.abspath(__file__))
absolute_project_root= os.path.abspath(os.path.join(current_dir,"../Phase_02_rgb/src/datasets"))

dataset= CIFAR10(
    root= absolute_project_root,
    train= True,
    download= False
)

# Get sample image
image, true_label= dataset[0]

#convert PIL---> numpy
image= np.array(image)

# convert HWC---> CHW
image= np.transpose(image,(2,0,1))
print("Input shape:", image.shape)

"""
--Convolution layer
"""
kernels = np.random.randn(8,3,3,3)

feature_maps= multi_filter_convolve(
    image= image,
    kernels= kernels,
    stride=1,
    padding=1
)
print ("After convolution:", feature_maps.shape)

#Applying relu
feature_maps= relu(feature_maps)
print("After ReLU:", feature_maps.shape)

# Pooling
pooled_maps= max_pool_multi_channel(
    feature_maps,
    pool_size=2,
    stride=2
)
print("After pooling:", pooled_maps.shape)

#flatten
flattened= flatten(pooled_maps)
print("After flatten:", flattened.shape)

# Dense layer
dense= DenseLayer(
    input_size= flattened.shape[0],
    output_size=10
)
logits= dense.forward(flattened)
print("Logits shape:", logits.shape)

#Applying softmax
probabilites= softmax(logits)

print("\n Class Probabilites:\n")

for i, prob in enumerate(probabilites):
    print(f"{classes[i]:{prob:4f}}")

#prediction
predicted_class= np.argmax(probabilites)

print("\nPredicted class:", classes[predicted_class])
print("True class:", classes[true_label])
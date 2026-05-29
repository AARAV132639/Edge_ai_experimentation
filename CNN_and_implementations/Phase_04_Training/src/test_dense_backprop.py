import numpy as np

import sys
import os
# This tells python to look two folder up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))

from Phase_03_classifier.src.dense import DenseLayer

dense= DenseLayer(
    input_size= 5,
    output_size=3
)

x= np.random.randn(5)

output= dense.forward(x)

print("Forward output:")
print(output)

gradient = np.array([
    0.1,
    -0.3,
    0.2
])

dW, dB= dense.backward(
    gradient,
    learning_rate=0.01
)

print("\n Weight gradient shape:")
print(dW.shape)

print("\n Bias gradient shape:")
print(dB.shape)
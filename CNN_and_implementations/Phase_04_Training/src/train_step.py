import numpy as np

import sys
import os
# This tells python to look two folder up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))

from Phase_03_classifier.src.dense import DenseLayer
from Phase_03_classifier.src.softmax import softmax

from entropy_loss import cross_entropy_loss
from gradients import softmax_crossentropy_gradient

dense = DenseLayer(
    input_size= 20,
    output_size=3
)

# Fake feature vector
x= np.random.randn(20)

# Assume correct class= 2
true_class =2
learning_rate =0.1

# Before training
logits= dense.forward(x)
probs= softmax(logits)

loss_before= cross_entropy_loss(
    probs,
    true_class 
)

print("Loss before:", loss_before)
print("Prob before:", probs)

## Back propogation
gradient = softmax_crossentropy_gradient(
    probs,
    true_class
)
dense.backward(
    gradient, learning_rate
)

# After update
logits= dense.forward(x)
probs= softmax(logits)

loss_after = cross_entropy_loss(
    probs,
    true_class 
)

print("Loss after:", loss_after)
print("Prob after:", probs) #Probability of correct class increased


"""

- Flow implemented:
                    i. make prediction
                    ii. measure mistake
                    iii. adjust itself
                    iv. make slightly better prediction

- File tip: Look into the mechanics and mathematics of this mini training loop again
"""
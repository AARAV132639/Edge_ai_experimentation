import numpy as np

def softmax_crossentropy_gradient(probabilities,true_class):
    """
    Computing dL/dz

    """
    target = np.zeros_like(probabilities)

    target[true_class]=1

    gradient= probabilities-target

    return gradient

if __name__=="__main__":
    probs= np.array([0.1,0.7,0.2])
    print(softmax_crossentropy_gradient(probs, 2))
    
"""
Look up this file again. concept yet not properly learnt
"""
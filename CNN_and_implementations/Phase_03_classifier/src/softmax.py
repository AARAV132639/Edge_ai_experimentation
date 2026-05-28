import numpy as np

def softmax(x):
    # Computing softmax probability

    # Numerical stability trick:
    x = x-np.max(x)

    exp_values= np.exp(x)

    probabiliteis= exp_values/np.sum(exp_values)
    return probabiliteis

if __name__=="__main__":
    sample= np.array([2.0,1.0,0.1])
    result= softmax(sample)

    print(result)
    print("Sum:", np.sum(result))

"""
1. Why are we subtracting max(x)?
    - This line prevents exp(1000) which would result in overflow numerically
    - This is standard deep learning practice

2. What softmax mathematically does?
    - Higher score--> exponentially larger probability

3. Why softmax is important?
    - Dense layer says---> class 3 seems strongest
    - Softmax says---> class has 78% confidence
"""
import numpy as np

def one_hot(class_index, num_classes):
    # Convert class index to one hot vector

    vector = np.zeros(num_classes)

    vector[class_index] =1

    return vector

if __name__=="__main__":
    print(one_hot(3,10))

"""

## Look up---> Why and what are we doing here? This concept is still not fully internalized

"""
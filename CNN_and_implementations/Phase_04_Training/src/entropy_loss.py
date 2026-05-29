import numpy as np

def cross_entropy_loss(predictions, true_class):
    """
    1. Computing cross entropy

    2. Parameters:
                i. Predictions: softmax probabilities
                ii. true_class: integer class label
    
    3. Returns: Scalar loss
    """

    epsilon = 1e-10

    predicted_probability= predictions[true_class]

    loss = -np.log(predicted_probability+ epsilon)

    return loss

if __name__ == "__main__":
    predictions = np.array([
        0.05,
        0.10,
        0.70,
        0.15
    ])
    true_class =3

    loss= cross_entropy_loss(
        predictions, true_class
    )
    
    print("Loss:", loss)

"""
1. Cross entropy heavily punishes confident mistakes which is exactly what we want during our training.

2. How cross entropy affects mathematically?

3. Why are we using epsilon?

"""
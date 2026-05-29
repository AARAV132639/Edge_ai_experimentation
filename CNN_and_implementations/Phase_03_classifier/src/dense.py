import numpy as np

class DenseLayer:

    def __init__(self, input_size, output_size):

        """
        Initialize weights and biases.

        - Input size---> number of input neurons
        - output size---> number of output neurons
        """

        self.weights = np.random.randn(output_size, input_size)*0.01 # Why this line exisits?
        self.biases = np.zeros((output_size,1))

        self.input= None # Why this change has been made?

    def forward(self,x):
            """
            Forward pass.

            - x shape---> (input_size,)
            - returns ---> (output_size,)
            """

            # Convert to column vector
            self.input= x.reshape(-1,1)

            output= np.dot(self.weights,self.input) + self.biases

            return output.flatten()
    
    def backward(self, dL_dz, learning_rate):
         
         dL_dz= dL_dz.reshape(-1,1)

         dL_dW = np.dot(dL_dz, self.input.T)

         dL_db= dL_dz

         # Gradient descent
         self.weights-= learning_rate*dL_dW
         self.biases-= learning_rate*dL_db

         return dL_dW, dL_db

if __name__=="__main__":
    sample_input = np.random.randn(2048)

    dense= DenseLayer(
        input_size= 2048,
        output_size= 10
    )

    result = dense.forward(sample_input)

    print("Output shape:", result.shape)
    print(result)

"""
## Insights

- Output shape:(10,) because CIFAR-10 has 10 classes

1. What is this layer doing mathematically?

    - The matrix (8,16,16) was flattened into 2048 features
    - Dense layer computes: Wx+b 
                                - W: learned weights(here random)
                                - x= feature vector
                                - b = biases
    - The final output got is Logits not probabilites

2. Why does this matter?
    - Convolution layers learn: local spatioal features
    - Dense laers combine them globally---> "edges+wheel texture+window shape=car"

"""
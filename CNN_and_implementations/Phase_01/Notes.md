# Implementation of Convolution, padding, stride handling and pooling operations from scratch

## Concept Questions

1. What does padding solves?
- Repeated convolution shrink images. Padding preserves spatial information

2. What is the concept of negative edges?

## Implementation done:
        1. Took a dataset from Fashion MNIST
        2. Applied convolution: applying a filter over an image vector
        3. In order to preserve the border signals/pixels we used padding
        4. On the feature map we used pooling: average pooling and max pooling
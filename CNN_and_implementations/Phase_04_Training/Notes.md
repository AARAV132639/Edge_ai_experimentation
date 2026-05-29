1. What training means?

- We want to answer:  Which weight caused the mistake, and by how much?

- For every weight. If Weight 17 contributed heavily to wrong prediction---> Decrease it
                    - If Weight 17 helped correct prediction----> increase it

---

2. Result case:

- For: Dense---> Softmax---> Cross entropy, the gradient simplifies dL/dz= p-y
        - p = predicted probabilities
        - y = one-hot encoded true label

- One of the most important results in deep learning.

---

3. Working

- Dense layer computes: z= Wx+b

- Backprop gives dL/dW= dL/dz xT and dl/db= dL/dz

- This means weight gradient = output error* input

- bias gradient = output error
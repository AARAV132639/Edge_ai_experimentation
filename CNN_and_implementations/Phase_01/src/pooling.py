import numpy as np

def max_pool2d (image, pool_size=2, stride=2):
    """
    Performing max pooling manually

    Paraemeters: 
        - image: input feature map(H,W)
        - pool_size: pooling window size
        - Stride: Movement step
    
    Returns:
        pooled output
    """

    image_h, image_w= image.shape

    output_h = ((image_h-pool_size)//stride)+1
    output_w= ((image_w- pool_size)//stride)+1

    output=  np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            vert_start= i*stride
            vert_end= vert_start+ pool_size
            
            horiz_start= j*stride
            horiz_end= horiz_start+ pool_size

            window= image[vert_start: vert_end, horiz_start: horiz_end]
            output[i,j] = np.max(window) #---> for max pooling
    
    return output

def average_pool2d(image, pool_size=2, stride=2):
    """
    Performing average pooling manually

    Parameters: 
        - Image: input feature map(H,W)
        - pool_size: pooling window size
        - stride: movement step
    
    Returns:
        - pooled output
    """
    image_h, image_w= image.shape

    output_h = ((image_h-pool_size)//stride)+1
    output_w= ((image_w- pool_size)//stride)+1

    output=  np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            vert_start= i*stride
            vert_end= vert_start+ pool_size
            
            horiz_start= j*stride
            horiz_end= horiz_start+ pool_size

            window= image[vert_start: vert_end, horiz_start: horiz_end]
            output[i,j] = np.mean(window) #---> for max pooling
    
    return output


if __name__== "__main__":
    sample= np.array([
        [1,3,2,4],
        [5,6,8,1],
        [0,9,2,3],
        [7,4,5,6]
    ])

    result= max_pool2d(sample)
    print(result)

    result = average_pool2d(sample)
    print (result)

import numpy as np

def max_pool_multi_channel(feature_maps, pool_size=2, stride=2):

    """
    Multi channel max pooling

    parameters: feature_maps(C,H,w)

    Returns: pooled feature maps
    
    """

    channels, height, width= feature_maps.shape

    output_h= ((height-pool_size)//stride)+1
    output_w= ((width-pool_size)//stride)+1

    output = np.zeros((channels, output_h, output_w))

    for c in range(channels):
        for i in range (output_h):
            for j in range(output_w):

                vert_start = i*stride
                vert_end = vert_start + pool_size

                horiz_start= j*stride
                horiz_end= horiz_start+ pool_size
                
                window = feature_maps[
                    c,
                    vert_start:vert_end,
                    horiz_start: horiz_end
                ]

                output[c,i,j]= np.max(window)
    
    return output
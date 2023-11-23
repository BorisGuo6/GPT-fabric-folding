import numpy as np
from PIL import Image

def save_depth_as_matrix(image_path, output_path="./test_depth.txt"):
    '''
    This function takes the path of the image and saves the depth image in a form where the background is 0
    We would pass this matrix to the LLM API in order to get the picking and placing pixels
    '''
    image = Image.open(image_path)
    image_array = np.array(image) / 255

    mask = image_array.copy()
    mask[mask > 0.646] = 0
    mask[mask != 0] = 1

    image_array = image_array * mask

    np.savetxt(output_path, np.round(image_array, decimals=2), fmt='%.2f')

test_image_path = "./eval result/DoubleTriangle/0/depth/2.png"
save_depth_as_matrix(test_image_path)
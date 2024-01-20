import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt 

def save_depth_as_matrix(image_path, output_path):
    '''
    This function takes the path of the image and saves the depth image in a form where the background is 0
    We would pass this matrix to the LLM API in order to get the picking and placing pixels
    '''
    image = Image.open(image_path)
    if image.size != 128:
        image = image.resize((128, 128))

    image_array = np.array(image) / 255

    mask = image_array.copy()
    mask[mask > 0.646] = 0
    mask[mask != 0] = 1

    image_array = image_array * mask
    image_array = image_array * 100
    np.savetxt(output_path, np.round(image_array, decimals=2), fmt='%.2f')

def find_corners(image_path):
    '''
    This function will use the OpenCV methods to detect the cloth corners from the given depth image
    '''
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10) 
    corners = np.int0(corners) 

    # Plotting the original image with the detected corners
    for i in corners: 
        x, y = i.ravel() 
        cv2.circle(img, (x, y), 3, 255, -1)     
    plt.imshow(img), plt.show() 
    plt.savefig("temp.png")

    return corners

# Replace 'your_image.jpg' with the actual path to your image file
image_path = './eval result/DoubleTriangle/0/depth/1.png'
cloth_corners = find_corners(image_path)

print("Pixel coordinates of cloth corners:")
for corner in cloth_corners:
    print(corner)

# test_input_path = "./eval result/DoubleTriangle/0/depth/1.png"
# test_goal_path = "./data/demo/DoubleTriangle/depth/2.png"
# save_depth_as_matrix(test_input_path, "./input_depth.txt")
# save_depth_as_matrix(test_goal_path, "./goal_depth.txt")
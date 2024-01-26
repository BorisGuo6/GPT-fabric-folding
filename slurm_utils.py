import numpy as np
from PIL import Image
import cv2
import os
from matplotlib import pyplot as plt 

def save_depth_as_matrix(image_path, output_path = None, save_matrix = True):
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
    if save_matrix:
        np.savetxt(output_path, np.round(image_array, decimals=2), fmt='%.2f')
    return image_array

def find_pixel_center_of_cloth(image_path):
    '''
    This function would be used to get the pixel center corresponding to the initial cloth configuration
    '''
    image_matrix = save_depth_as_matrix(image_path, None, False)

    # Find indices of non-zero values
    nonzero_indices = np.nonzero(image_matrix)

    # Calculate the center coordinates
    center_x = int(np.mean(nonzero_indices[1]))
    center_y = int(np.mean(nonzero_indices[0]))

    return (center_x, center_y)

def find_corners(image_path):
    '''
    This function will use the OpenCV methods to detect the cloth corners from the given depth image
    '''
    image_matrix = save_depth_as_matrix(image_path, None, False)
    cv2.imwrite("./to_be_deleted.png", image_matrix)

    img = cv2.imread("./to_be_deleted.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Implementing Harris corner detection by myself to get the corners. Does not work though
    # corners = cv2.cornerHarris(gray,2,3,0.04)
    # corners_thresholded = corners > 0.01 * corners.max()
    # corner_coordinates = np.array(np.where(corners_thresholded)).T
    
    # Using OpenCV.goodFeaturesToTrack() function to get the corners
    corner_coordinates = cv2.goodFeaturesToTrack(image = gray, maxCorners = 27, qualityLevel = 0.04, minDistance = 10, useHarrisDetector = True) 
    corner_coordinates = np.intp(corner_coordinates) 

    # Plotting the original image with the detected corners
    if __name__ == "__main__":
        for i in corner_coordinates: 
            x, y = i.ravel() 
            cv2.circle(img, (x, y), 3, 255, -1)     
        plt.imshow(img), plt.show() 
        plt.savefig("temp.png")

    os.remove("./to_be_deleted.png")
    return corner_coordinates

if __name__ == "__main__":
    # Replace 'your_image.jpg' with the actual path to your image file
    image_path = './eval result/DoubleTriangle/3/depth/1.png'
    cloth_corners = find_corners(image_path)

    print("Pixel coordinates of cloth center:", find_pixel_center_of_cloth(image_path))

    print("Pixel coordinates of cloth corners:")
    for corner in cloth_corners:
        print(corner)
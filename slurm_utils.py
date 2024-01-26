import numpy as np
from PIL import Image
import cv2
import os
from matplotlib import pyplot as plt 
import pickle
from pprint import pprint

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

def analyze_foldsformer_pickles(pickle_path):
    '''
    DO NOT import this. I am simply playing around with this function lol
    '''
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    pprint(data)

def get_mean_particle_distance_error(eval_dir, expert_dir, cached_path, cherry_pick = False):
    '''
    This function is used to generate the mean particle distance error between the eval and expert results
    When the cherry pick boolean is set to True, we only get the numbers for the success scenarios for a given experimental run
    '''
    # Manually inspected success cases for the most recent run for DoubleTriangle (2024-01-26.log)
    successful_indices = [0,1,2,4,6,8,16,17,19,21,22,26,32,38]

    # Get the number of configs on which are we experimenting (could be hard-coded to 40)
    total_indices_len = 0
    with open(cached_path, "rb") as f:
        _, init_states = pickle.load(f)
        total_indices_len = len(init_states)
    total_indices = [i for i in range(total_indices_len)]    

    # Based on cherry picking, decide the indices to be picked
    test_indices = successful_indices if cherry_pick else total_indices

    # Now actually go through each and every saved final cloth configuration and compute the distances
    distance_list = []
    for config_id in test_indices:
        eval_info = os.path.join(eval_dir, str(config_id), "info.pkl")
        expert_info = os.path.join(expert_dir, str(config_id), "info.pkl")
        with open(eval_info, "rb") as f:
            eval_pos = pickle.load(f)
        with open(expert_info, "rb") as f:
            expert_pos = pickle.load(f)
        eval_pos = eval_pos['pos']
        expert_pos = expert_pos['pos']

        # [TODO] Perform some pre-processing to make sure that the inclinations align
        dist = np.linalg.norm(expert_pos - eval_pos, axis=1).mean()
        distance_list.append(dist)

    # Get the mean of the mean of the overall distance_list that we have
    return sum(distance_list) * 1. / len(distance_list)

if __name__ == "__main__":
    mean_err = get_mean_particle_distance_error("eval result/DoubleTriangle/square", "data/demonstrations/DoubleTriangle/square", "cached configs/square.pkl")
    print(mean_err)
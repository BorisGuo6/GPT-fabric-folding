import numpy as np
from PIL import Image
import cv2
import os
from matplotlib import pyplot as plt 
import pickle

def save_depth_as_matrix(image_path, output_path = None, save_matrix = True, should_crop = True):
    '''
    This function takes the path of the image and saves the depth image in a form where the background is 0
    We would pass this matrix to the LLM API in order to get the picking and placing pixels
    Note that this function will try to crop the given input image to 128 x 128 if we pass the should_crop param
    '''
    image = Image.open(image_path)
    if should_crop:
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

def find_pixel_center_of_cloth(image_path, should_crop = True):
    '''
    This function would be used to get the pixel center corresponding to the initial cloth configuration
    '''
    image_matrix = save_depth_as_matrix(image_path, None, False, should_crop)

    # Find indices of non-zero values
    nonzero_indices = np.nonzero(image_matrix)

    # Calculate the center coordinates
    center_x = int(np.mean(nonzero_indices[1]))
    center_y = int(np.mean(nonzero_indices[0]))

    return (center_x, center_y)

def find_corners(image_path, should_crop = True):
    '''
    This function will use the OpenCV methods to detect the cloth corners from the given depth image
    '''
    image_matrix = save_depth_as_matrix(image_path, None, False, should_crop)
    cv2.imwrite("./to_be_deleted.png", image_matrix)

    img = cv2.imread("./to_be_deleted.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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

def get_mean_particle_distance_error(eval_dir, expert_dir, cached_path, task, config_id):
    '''
    This function is used to generate the mean particle distance error between the eval and expert results
    '''
    # Get the number of configs on which are we experimenting (could be hard-coded to 40)
    total_indices_len = 0
    with open(cached_path, "rb") as f:
        _, init_states = pickle.load(f)
        total_indices_len = len(init_states)
    total_indices = [i for i in range(total_indices_len)]

    # We pass the config ID to get the number while calling this function from the evaluation script 
    if config_id == None:
        test_indices = total_indices
    else:
        test_indices = [config_id]

    # Now actually go through each and every saved final cloth configuration and compute the distances
    distance_list = []

    # Number of possible configurations for the given kind of fold. 
    if task == "DoubleTriangle":
        num_info = 8
    elif task == "AllCornersInward":
        num_info = 9
    else:
        num_info = 16

    for config_id in test_indices:
        eval_info = os.path.join(eval_dir, str(config_id), "info.pkl")
        with open(eval_info, "rb") as f:
            eval_pos = pickle.load(f)
        eval_pos = eval_pos['pos']

        min_dist = np.inf
        for i in range(num_info):
            expert_info = os.path.join(expert_dir, str(config_id), "info-" + str(i) + ".pkl")
            with open(expert_info, "rb") as f:
                expert_pos = pickle.load(f)

            expert_pos = expert_pos['pos']
            min_dist = min(min_dist, np.linalg.norm(expert_pos - eval_pos, axis=1).mean())
        distance_list.append(min_dist)

    return sorted(distance_list)

def merge_images_horizontally(parent_path):
    '''
    DO NOT import this. It's just a helper function to merge images horizontally
    '''
    num_images = 4
    img_list = []
    for i in range(num_images):
        file_path = os.path.join(parent_path, "rgbviz", str(i) + ".png")
        img = cv2.imread(file_path)
        img_list.append(img)
    merged_image = np.concatenate(img_list, axis = 1)
    write_path = os.path.join(parent_path, "rgbviz", "merged.png")
    cv2.imwrite(write_path, merged_image)

def get_test_run_stats(parent_eval_dir, parent_expert_dir, cached_path, task):
    '''
    This keeps calling the script to get the mean particle distance error multiple times for the given config Id
    '''
    num_configs = 40
    num_tests = 2
    all_scores = np.zeros((num_tests, num_configs))
    for test in range(num_tests):
        for config in range(num_configs):
            eval_dir = os.path.join(parent_eval_dir, str(test))
            score = get_mean_particle_distance_error(eval_dir, parent_expert_dir, cached_path, task, config)
            all_scores[test, config] = score[0]
    min_list = np.zeros(num_configs)
    avg_list = np.zeros(num_configs)
    for config in range(num_configs):
        min_list[config] = np.min(all_scores[:, config])
        avg_list[config] = np.mean(all_scores[:, config])
    
    # Printing the stats reported
    print("Mean and Std dev for the min values(mm): ", np.mean(min_list) * 1000, np.std(min_list) * 1000)
    print("Mean and Std dev for the mean values(mm): ", np.mean(avg_list) * 1000, np.std(avg_list) * 1000)

if __name__ == "__main__":
    get_test_run_stats("eval result/CornersEdgesInward/square/2024-02-26", "data/demonstrations/CornersEdgesInward/square", "cached configs/square.pkl", "CornersEdgesInward")
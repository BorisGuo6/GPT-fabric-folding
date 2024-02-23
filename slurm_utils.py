import numpy as np
from PIL import Image
import cv2
import os
from matplotlib import pyplot as plt 
import pickle
from pprint import pprint
from scipy.optimize import minimize, differential_evolution

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
    mat_data = data['pos']
    print(mat_data.shape)
    print(np.min(mat_data[:, 0]), np.max(mat_data[:, 0]))
    print(np.min(mat_data[:, 1]), np.max(mat_data[:, 1]))
    print(np.min(mat_data[:, 2]), np.max(mat_data[:, 2]))

def get_mean_particle_distance_error(eval_dir, expert_dir, cached_path, task, config_id):
    '''
    This function is used to generate the mean particle distance error between the eval and expert results
    '''
    def rotate_anticlockwise(expert_pos):
        ## Defining this function to rotate the expert position by 90% in anticlockwise direction
        ## Note that the 0 column is the x-axis, 1 column is the z-axis, and 2 column is the y-axis
        ## The flow would look like: (x,z,y) -> (y, z, -x) -> (-x, z, -y) -> (-y, z, x) -> (x, z, y)
        for i in range(len(expert_pos)):
            x, z, y = expert_pos[i,0], expert_pos[i,1], expert_pos[i,2]
            expert_pos[i,0] = y
            expert_pos[i,1] = z
            expert_pos[i,2] = x * -1.0
        return expert_pos

    def correct_z_alignment(A, B):
        # Objective function to minimize (sum of Euclidean distances)
        def objective_function(x, A, B):
            if len(np.unique(x)) < len(x):
                # Penalize non-unique indices
                penalty = 1e6
                return penalty
            A_permuted = A[x.astype(int)]
            distances = np.linalg.norm(A_permuted - B, axis=1)
            return np.sum(distances)
        
        # Having bounds on the values possible for the indices
        bounds = [(0, len(A)-1)] * len(A)

        # Minimize the objective function
        result = differential_evolution(objective_function, bounds, args=(A, B))

        # Get the optimal permutation indices
        optimal_permutation = result.x.astype(int)

        # Permute the rows in A and B based on the optimal permutation
        A_permuted = A[optimal_permutation]
        B_permuted = B

        return np.linalg.norm(A_permuted - B_permuted, axis=1).mean()

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
        # print(config_id, min_dist)
        distance_list.append(min_dist)

    return sorted(distance_list)

def merge_images_horizontally(parent_path):
    '''
    DO NOT import this. It's just a helper function to merge images horizontally
    '''
    num_images = 5
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
    num_tests = 5
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
    print("Mean and Std dev for the min values: ", np.mean(min_list) * 1000, np.std(min_list) * 1000)
    print("Mean and Std dev for the mean values: ", np.mean(avg_list) * 1000, np.std(avg_list) * 1000)

if __name__ == "__main__":
    get_test_run_stats("eval result/AllCornersInward/square/2024-02-21", "data/demonstrations/AllCornersInward/square", "cached configs/square.pkl", "AllCornersInward")
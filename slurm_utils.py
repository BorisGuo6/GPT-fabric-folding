import numpy as np
from PIL import Image
import cv2
import os
from matplotlib import pyplot as plt 
import pickle
from pprint import pprint
from scipy.optimize import minimize, differential_evolution
from utils.gpt_utils import analyze_images_gpt, get_user_prompt, system_prompt, gpt_v_demonstrations, parse_output
import json
from openai import OpenAI

def save_depth_as_matrix(image_path, output_path = None, save_matrix = True, should_crop = True):
    '''
    This function takes the path of the image and saves the depth image in a form where the background is 0
    We would pass this matrix to the LLM API in order to get the picking and placing pixels
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
    print("Mean and Std dev for the min values: ", np.mean(min_list) * 1000, np.std(min_list) * 1000)
    print("Mean and Std dev for the mean values: ", np.mean(avg_list) * 1000, np.std(avg_list) * 1000)

### These functions are all corresponding to the helper functions that we'd use for the real world experiments
def crop_input_image(input_rgb, input_depth, cropped_depth_image):
    '''
    This function call will be used for taking the input RGB and Depth images to be cropped
    The final saved image should be a depth image with the cloth around the center
    Note that this function call returns the pivot pixel coordinate for handling the real pick, place pixels
    '''
    # Load the image
    image = cv2.imread(input_rgb)
    depth_image = cv2.imread(input_depth)
    pivot_coordinate = np.array([0, 0])

    # Crop the image initially since there's a big bounding box present already
    height, width = image.shape[:2]
    image = image[20:height-20, 20:width-20]
    depth_image = depth_image[20:height-20, 20:width-20]
    pivot_coordinate += np.array([20, 20])

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of the image
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    # Calculate the distances of contours from the center
    distances = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            distance = np.sqrt((cX - center_x) ** 2 + (cY - center_y) ** 2)
            distances.append((contour, distance))

    # Sort the contours by distance from the center
    distances.sort(key=lambda x: x[1])

    # Select the contour closest to the center as the region of interest (ROI)
    closest_contour = distances[0][0]

    # Get the bounding box of the closest contour
    x, y, w, h = cv2.boundingRect(closest_contour)

    # Add padding around the bounding box
    padding = 40
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Ensure the bounding box stays within the image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(image.shape[1] - x, w)
    h = min(image.shape[0] - y, h)

    # Crop the region of interest with padding
    cropped_image = image[y:y+h, x:x+w]
    image[y,x] = (255, 0, 0)
    pivot_coordinate += np.array([x, y])

    # Display the cropped image to see if it's all working alright :)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Now instead performing cropping with the depth image
    depth_image = depth_image[y:y+h, x:x+w]
    cv2.imwrite(cropped_depth_image, depth_image)

    # Return the pivot pixel coordinates to be used later
    return pivot_coordinate

def get_initial_cloth_center(initial_input_rgb, initial_input_depth, initial_cropped_depth_image):
    '''
    This function takes the RGB and Depth images of the initial cloth configuration and returns the actual pixel coordinate for the center
    '''
    # First crop the initial cloth images to get the new depth image
    initial_pivot_coordinate = crop_input_image(initial_input_rgb, initial_input_depth, initial_cropped_depth_image)

    # Call the function to get the center for the saved initial cropped depth image
    initial_local_cloth_center = find_pixel_center_of_cloth(initial_cropped_depth_image, False)

    # Add the pivot coordinate to return the true coordinates of the cloth center
    return initial_pivot_coordinate + np.array(initial_local_cloth_center)

def gpt_for_real_world(input_rgb, input_depth, cropped_depth_image, cloth_center, task, current_step):
    '''
    This function call will be used for calling the overall GPT pipeline for the real-world experiments
    Args:
        input_rgb: The file path corresponding to the RGB image of the current cloth configuration
        input_depth: The file path corresponding to the depth image of the current cloth configuration
        cropped_depth_image: The file path where the cropped depth image is expected to be saved
        cloth_center: The pixel coordinates for the initial cloth center in the actual image
        task: The folding task that we wish to perform. Use one of DoubleTriangle, AllCornersInward, CornersEdgesInward, DoubleStraight
        current_step: The number of folding steps executed for the current test case thus far (starts with 0)
    '''
    # Setting up the chain to interact with OpenAI. Using Daniel's API key for now
    client = OpenAI(api_key="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")

    # Crop the input RGB and Depth images to get the cropped version of theirs
    pivot_coordinate = crop_input_image(input_rgb, input_depth, cropped_depth_image)

    # Get the local cloth center for the current image configuration
    cloth_center = cloth_center - pivot_coordinate

    # Get the cloth corners for the current cloth configuration
    cloth_corners = find_corners(cropped_depth_image, False)

    # Getting the template folding instruction images from the demonstrations
    demo_root_path = os.path.join("data", "demo", task, "rgbviz")
    start_image = os.path.join(demo_root_path, str(current_step) + ".png")
    last_image = os.path.join(demo_root_path, str(current_step+1) + ".png")
    instruction = analyze_images_gpt([start_image, last_image], task, current_step)

    # Getting the user prompt based on the information that we have so far
    user_prompt = get_user_prompt(cloth_corners, cloth_center, True, instruction, task, None)
    print("The user prompt was: ", user_prompt)

    # Getting the demonstrations that would be used for the specific task
    indices = gpt_v_demonstrations[task]["gpt-demonstrations"]

    # Imp: The information corresponding to the demonstrations is assumed to be in utils folder
    demonstration_dictionary_list = []
    gpt_demonstrations_path = os.path.join("utils", "gpt-demonstrations", task, "demonstrations.json")
    with open(gpt_demonstrations_path, 'r') as f:
        gpt_demonstrations = json.load(f)

    # Fetching the information from the demonstrations as In-context data
    for index in indices:
        step_dictionary = gpt_demonstrations[str(index)][str(current_step + 1)]
        user_prompt_dictionary = {
            "role": "user",
            "content": step_dictionary["user-prompt"]
        }
        assistant_response_dictionary = {
            "role": "assistant",
            "content": step_dictionary["assistant-response"]
        }
        demonstration_dictionary_list += [user_prompt_dictionary, assistant_response_dictionary]
    
    # Making a call to the OpenAI API after we have the demonstrations
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            }] + demonstration_dictionary_list +
            [{
                "role": "user",
                "content": user_prompt
            }],
        temperature=0,
        max_tokens=769,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Parsing the above output to get the pixel coorindates for pick and place
    test_pick_pixel, test_place_pixel = parse_output(response.choices[0].message.content)
    print("The system response was: ", response.choices[0].message.content)

    # Returning the pick and the place point after adjusting with the pivot coordinate
    return pivot_coordinate + test_pick_pixel, pivot_coordinate + test_place_pixel

if __name__ == "__main__":
    get_test_run_stats("eval result/CornersEdgesInward/square/2024-02-26", "data/demonstrations/CornersEdgesInward/square", "cached configs/square.pkl", "CornersEdgesInward")
    # Remember this image for Double Straight - eval result/DoubleStraight/rectangle/2024-02-23/4/7
    # merge_images_horizontally("eval result/DoubleStraight/rectangle/2024-02-26/4/29")
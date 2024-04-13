import numpy as np
import cv2
import os
from utils.gpt_utils import analyze_images_gpt, get_user_prompt, system_prompt, gpt_v_demonstrations, parse_output
import json
from slurm_utils import find_pixel_center_of_cloth, find_corners
from openai import OpenAI

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
    # TODO: Remove the API keys before releasing the code in public
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
    instruction = analyze_images_gpt([start_image, last_image], task, current_step, "in-context")

    # Getting the user prompt based on the information that we have so far
    user_prompt = get_user_prompt(cloth_corners, cloth_center, True, instruction, task, None)
    print("The user prompt was: ", user_prompt)

    # Getting the demonstrations that would be used for the specific task
    indices = gpt_v_demonstrations["in-context"][task]["gpt-demonstrations"]

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
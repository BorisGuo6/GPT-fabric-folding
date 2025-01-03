import argparse
import numpy as np
from utils.visual import get_pixel_coord_from_world, get_world_coord_from_pixel
import pyflex
import os
from tqdm import tqdm
import imageio
import pickle
from utils.visual import action_viz
from softgym.envs.foldenv import FoldEnv
from Demonstrator.demonstrator import Demonstrator
from slurm_utils import find_corners, find_pixel_center_of_cloth
from utils.gpt_utils import system_prompt, get_user_prompt
import json

def main():
    parser = argparse.ArgumentParser(description="Generate Demonstrations")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument("--img_size", type=int, default=128, help="Size of rendered image")
    parser.add_argument("--cached", type=str, help="Cached filename")
    args = parser.parse_args()

    # Getting the path to the root directory
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    parent_directory = os.path.dirname(script_directory)

    # env settings
    cached_path = os.path.join(parent_directory, "cached configs", args.cached + ".pkl")
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # demonstrator settings
    demonstrator = Demonstrator[args.task]()

    # save settings
    save_path = os.path.join(parent_directory, "data", "gpt-demonstrations", args.task, args.cached, str(args.img_size))
    os.makedirs(save_path, exist_ok=True)

    # other settings
    rgb_shape = (args.img_size, args.img_size)
    num_data = env.num_configs

    # The json file corresponding to all the demonstrations
    instructions_json = {}

    for config_id in tqdm(range(num_data)):
        # Creating a new json dict for the current config ID
        instructions_json[config_id] = {}

        # folders
        save_folder = os.path.join(save_path, str(config_id))
        save_folder_rgb = os.path.join(save_folder, "rgb")
        save_folder_depth = os.path.join(save_folder, "depth")
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_folder_rgb, exist_ok=True)
        os.makedirs(save_folder_depth, exist_ok=True)

        pick_pixels = []
        place_pixels = []
        rgbs = []

        # env reset
        env.reset(config_id=config_id)
        camera_params = env.camera_params
        rgb, depth = env.render_image()
        imageio.imwrite(os.path.join(save_folder_rgb, str(0) + ".png"), rgb)
        depth = depth * 255
        depth = depth.astype(np.uint8)
        imageio.imwrite(os.path.join(save_folder_depth, str(0) + ".png"), depth)
        rgbs.append(rgb)
        cloth_center = find_pixel_center_of_cloth(os.path.join(save_folder_depth, str(0) + ".png"))

        if args.task == "DoubleTriangle":
            for i in range(3):
                # Getting a json object corresponding to each step
                step_json = {}

                # Detecting corners for the current cloth configuration
                image_path = os.path.join(save_folder_depth, str(i) + ".png")
                cloth_corners = find_corners(image_path)

                # Getting the user prompt and saving it in the json file
                user_prompt = get_user_prompt(cloth_corners, cloth_center, False, "", args.task, i)
                step_json["user-prompt"] = user_prompt
                step_json["system-prompt"] = system_prompt

                ### Now getting the expected assistant's response by performing logical operations on top of this
                thought_process = "Thought Process:\n"

                if len(cloth_corners) == 2:
                    thought_process += "The provided method of folding indicates that the cloth should be folded by bringing the most distant corners together.\n"
                    thought_process += "Now since there are only two corners provided to us, one of them will be the pick point and the other one will be the place point.\n\n"
                    pick_idx = 0
                    place_idx = 1
                    thought_process += "By picking the cloth at " + str(tuple(cloth_corners[pick_idx][0])) + " and placing it at the corner at " + str(tuple(cloth_corners[place_idx][0])) + " the resulting fold aligns with the instructions provided."
                else:
                    # First step - Just print generic stuff into it that could be used later. State that we need to select the top three farthest corners
                    thought_process += "The provided method of folding indicates that the cloth should be folded by bringing the most distant corners together.\n\n"
                    thought_process += "There is a list of corners provided to me and I have to identify the point to be picked among these corners. To do so, I will first get the three corners that are the farthest from the center. The picking point and placing point will be one of these.\n\n"
                    thought_process += "I will use the square of the Euclidean distance between each corner and the center, " + str(cloth_center) +  ", and select the corners with the maximum distance.\n"

                    # Second step - Now, go through all the cloth corners and get the corner that is the farthest from the center
                    for (j, corner) in enumerate(cloth_corners):
                        corner = corner[0]
                        thought_process += str(j) + ": The distance for " + str(tuple(corner)) + "is "
                        thought_process += "(" + str(corner[0]) + "-" + str(cloth_center[0]) + ")^2 + (" + str(corner[1]) + "-" + str(cloth_center[1]) + ")^2 = "
                        thought_process += str(int(np.linalg.norm(corner - cloth_center) ** 2)) + "\n"
                    distances = [np.linalg.norm(x - cloth_center) for x in cloth_corners]
                    distances = list(enumerate(distances))
                    distances = sorted(distances, key = lambda x : x[1], reverse=True)
                    indices = [index for index, _ in distances[:3]]
                    thought_process += "\nFrom this list above, we see that the three cloth corners that are the farthest from the center are:\n"
                    for index in indices:
                        thought_process += str(tuple(cloth_corners[index][0])) + "\n"

                    cloth_corners = cloth_corners[indices]
                    print(cloth_corners)
                    # Third step - Now get the three possible pairs of combinations and get the pair with max distance
                    thought_process += "\nTo decide the pick and place point, I will compute the distances across each pair of points among these three points above and get the pair with the highest distance. I will again use the square of Euclidean distance here.\n"
                    thought_process += str(0) + ": The distance between the first and the second point is "
                    thought_process += "(" + str(cloth_corners[0][0][0]) + "-" + str(cloth_corners[1][0][0]) + ")^2 + (" + str(cloth_corners[0][0][1]) + "-" + str(cloth_corners[1][0][1]) + ")^2 = "
                    thought_process += str(int(np.linalg.norm(cloth_corners[0][0] - cloth_corners[1][0]) ** 2)) + "\n"
                    dist_1 = np.linalg.norm(cloth_corners[0][0] - cloth_corners[1][0])

                    thought_process += str(1) + ": The distance between the second and the third point is "
                    thought_process += "(" + str(cloth_corners[2][0][0]) + "-" + str(cloth_corners[1][0][0]) + ")^2 + (" + str(cloth_corners[2][0][1]) + "-" + str(cloth_corners[1][0][1]) + ")^2 = "
                    thought_process += str(int(np.linalg.norm(cloth_corners[2][0] - cloth_corners[1][0]) ** 2)) + "\n"
                    dist_2 = np.linalg.norm(cloth_corners[2][0] - cloth_corners[1][0])

                    thought_process += str(2) + ": The distance between the first and the third point is "
                    thought_process += "(" + str(cloth_corners[0][0][0]) + "-" + str(cloth_corners[2][0][0]) + ")^2 + (" + str(cloth_corners[0][0][1]) + "-" + str(cloth_corners[2][0][1]) + ")^2 = "
                    thought_process += str(int(np.linalg.norm(cloth_corners[0][0] - cloth_corners[2][0]) ** 2)) + "\n"
                    dist_3 = np.linalg.norm(cloth_corners[0][0] - cloth_corners[2][0])

                    pick_idx, place_idx = -1, -1
                    if dist_1 > dist_2 and dist_1 > dist_3:
                        pick_idx, place_idx = 0, 1
                    elif dist_2 > dist_1 and dist_2 > dist_3:
                        pick_idx, place_idx = 1, 2
                    else:
                        pick_idx, place_idx = 0, 2

                    thought_process += "\nHere we see that the points that are the most distant are the point " + str(pick_idx) + " and the point " + str(place_idx) + "."
                    thought_process += "Thus, by selecting these points as the pick and the place points, we achieve a diagonal fold.\n"
                    thought_process += "By picking the cloth at " + str(tuple(cloth_corners[pick_idx][0])) + " and placing it at the corner at " + str(tuple(cloth_corners[place_idx][0])) + " the resulting fold aligns with the instructions provided."
                
                test_pick_pixel = np.array(cloth_corners[pick_idx][0])
                test_place_pixel = np.array(cloth_corners[place_idx][0])

                # Now saving these coorindates in the planning string as well
                planning_string = "Planning:\n"
                planning_string += "Pick Point = " + "(" + str(test_pick_pixel[0]) + ", " + str(test_pick_pixel[1]) + ")\n"
                planning_string += "Place Point = " + "(" + str(test_place_pixel[0]) + ", " + str(test_place_pixel[1]) + ")\n"
                step_json["assistant-response"] = planning_string + "\n" + thought_process
                print(planning_string + "\n" + thought_process)
                instructions_json[config_id][str(i)] = step_json

                test_pick_pixel = tuple(test_pick_pixel)
                test_place_pixel = tuple(test_place_pixel)
                print("The Pick and the place pixels", test_pick_pixel, test_place_pixel)

                pick_pixels.append(test_pick_pixel)
                place_pixels.append(test_place_pixel)

                # convert the pixel cords into world cords
                test_pick_pos = get_world_coord_from_pixel(test_pick_pixel, depth, camera_params)
                test_place_pos = get_world_coord_from_pixel(test_place_pixel, depth, camera_params)

                # pick & place
                env.pick_and_place(test_pick_pos.copy(), test_place_pos.copy())

                # render & update frames & save
                rgb, depth = env.render_image()
                depth_save = depth.copy() * 255
                depth_save = depth_save.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth_save)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                rgbs.append(rgb)

        elif args.task == "AllCornersInward":
            for i in range(5):
                # Getting a json object corresponding to each step
                step_json = {}

                # Detecting corners for the current cloth configuration
                image_path = os.path.join(save_folder_depth, str(i) + ".png")
                cloth_corners = find_corners(image_path)

                # Getting the user prompt and saving it in the json file
                user_prompt = get_user_prompt(cloth_corners, cloth_center, False, "", args.task, i)
                step_json["user-prompt"] = user_prompt
                step_json["system-prompt"] = system_prompt

                ### Now getting the expected assistant's response by performing logical operations on top of this
                thought_process = "Thought Process:\n"

                # First step - Just print generic stuff into it that could be used later. State that we need to selected the farthest corner
                thought_process += "The provided method of folding indicates that the cloth should be folded by bringing a corner to the center of the cloth.\n\n"
                thought_process += "There is a list of corners provided to me and I have to identify the point to be picked among these corners. To do so, I will select the corner point that is the farthest from the given cloth center in order to achieve the maximum fold.\n\n"
                thought_process += "I will use the square of the Euclidean distance between each corner and the center, " + str(cloth_center) +  ", and select the corner with the maximum distance.\n"

                # Second step - Now, go through all the cloth corners and get the corner that is the farthest from the center
                for (j, corner) in enumerate(cloth_corners):
                    corner = corner[0]
                    thought_process += str(j) + ": The distance for " + str(tuple(corner)) + "is "
                    thought_process += "(" + str(corner[0]) + "-" + str(cloth_center[0]) + ")^2 + (" + str(corner[1]) + "-" + str(cloth_center[1]) + ")^2 = "
                    thought_process += str(int(np.linalg.norm(corner - cloth_center) ** 2)) + "\n"
                distances = [np.linalg.norm(x - cloth_center) for x in cloth_corners]
                index = np.argmax(distances)
                thought_process += "From this list above, we see that the cloth corner that is the farthest from the center is " + str(tuple(cloth_corners[index][0])) + "\n\n"

                # Third step - Now select the above point as the pick point and the center as the place point and save it to the response
                thought_process += "By picking the cloth at " + str(tuple(cloth_corners[index][0])) + " and placing it at the center at " + str(cloth_center) + " the resulting fold aligns with the instructions provided."

                # Now saving these coorindates in the planning string as well
                planning_string = "Planning:\n"
                planning_string += "Pick Point = " + "(" + str(cloth_corners[index][0][0]) + ", " + str(cloth_corners[index][0][1]) + ")\n"
                planning_string += "Place Point = " + "(" + str(cloth_center[0]) + ", " + str(cloth_center[1]) + ")\n"
                step_json["assistant-response"] = planning_string + "\n" + thought_process
                print(planning_string + "\n" + thought_process)
                instructions_json[config_id][str(i)] = step_json

                test_pick_pixel = tuple(cloth_corners[index][0])
                test_place_pixel = cloth_center
                print("The Pick and the place pixels", test_pick_pixel, test_place_pixel)

                pick_pixels.append(test_pick_pixel)
                place_pixels.append(test_place_pixel)

                # convert the pixel cords into world cords
                test_pick_pos = get_world_coord_from_pixel(test_pick_pixel, depth, camera_params)
                test_place_pos = get_world_coord_from_pixel(test_place_pixel, depth, camera_params)

                # pick & place
                env.pick_and_place(test_pick_pos.copy(), test_place_pos.copy())

                # render & update frames & save
                rgb, depth = env.render_image()
                depth_save = depth.copy() * 255
                depth_save = depth_save.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth_save)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                rgbs.append(rgb)

        elif args.task == "CornersEdgesInward":
            for i in range(5):
                # Getting a json object corresponding to each step
                step_json = {}

                # Detecting corners for the current cloth configuration
                image_path = os.path.join(save_folder_depth, str(i) + ".png")
                cloth_corners = find_corners(image_path)

                # Getting the user prompt and saving it in the json file
                user_prompt = get_user_prompt(cloth_corners, cloth_center, False, "", args.task, i)
                step_json["user-prompt"] = user_prompt
                step_json["system-prompt"] = system_prompt

                ### Now getting the expected assistant's response by performing logical operations on top of this
                thought_process = "Thought Process:\n"

                # Fold the corners inward for the first two steps
                if i < 3:
                    # First step - Just print generic stuff into it that could be used later. State that we need to selected the farthest corner
                    thought_process += "The provided method of folding indicates that the cloth should be folded by bringing a corner to the center of the cloth.\n\n"
                    thought_process += "There is a list of corners provided to me and I have to identify the point to be picked among these corners. To do so, I will select the corner point that is the farthest from the given cloth center in order to achieve the maximum fold.\n\n"
                    thought_process += "I will use the square of the Euclidean distance between each corner and the center, " + str(cloth_center) +  ", and select the corner with the maximum distance.\n"

                    # Second step - Now, go through all the cloth corners and get the corner that is the farthest from the center
                    for (j, corner) in enumerate(cloth_corners):
                        corner = corner[0]
                        thought_process += str(j) + ": The distance for " + str(tuple(corner)) + "is "
                        thought_process += "(" + str(corner[0]) + "-" + str(cloth_center[0]) + ")^2 + (" + str(corner[1]) + "-" + str(cloth_center[1]) + ")^2 = "
                        thought_process += str(int(np.linalg.norm(corner - cloth_center) ** 2)) + "\n"
                    distances = [np.linalg.norm(x - cloth_center) for x in cloth_corners]
                    index = np.argmax(distances)
                    thought_process += "From this list above, we see that the cloth corner that is the farthest from the center is " + str(tuple(cloth_corners[index][0])) + "\n\n"

                    # Third step - Now select the above point as the pick point and the center as the place point and save it to the response
                    thought_process += "By picking the cloth at " + str(tuple(cloth_corners[index][0])) + " and placing it at the center at " + str(cloth_center) + " the resulting fold aligns with the instructions provided."

                    # Now saving these coorindates in the planning string as well
                    planning_string = "Planning:\n"
                    planning_string += "Pick Point = " + "(" + str(cloth_corners[index][0][0]) + ", " + str(cloth_corners[index][0][1]) + ")\n"
                    planning_string += "Place Point = " + "(" + str(cloth_center[0]) + ", " + str(cloth_center[1]) + ")\n"
                    step_json["assistant-response"] = planning_string + "\n" + thought_process
                    print(planning_string + "\n" + thought_process)
                    instructions_json[config_id][str(i)] = step_json

                    # Print the test pick and place pixels
                    test_pick_pixel = tuple(cloth_corners[index][0])
                    test_place_pixel = cloth_center

                # For the next two steps, pick the farthest point from the center and fold it towards the point that is the closest to it
                else:
                    # First step - Just print generic stuff into it that could be used later. State that we need to selected the farthest corner
                    thought_process += "The provided method of folding indicates that the cloth should be folded by bringing a corner to its nearest edge.\n\n"
                    thought_process += "There is a list of corners provided to me and I have to identify the point to be picked among these corners. To do so, I will first get the four corners that are the farthest from the center. The picking point and placing point will be one of these.\n\n"
                    thought_process += "I will use the square of the Euclidean distance between each corner and the center, " + str(cloth_center) +  ", and select the corners with the maximum distance.\n"

                    # Second step - Now, go through all the cloth corners and get the corner that is the farthest from the center
                    for (j, corner) in enumerate(cloth_corners):
                        corner = corner[0]
                        thought_process += str(j) + ": The distance for " + str(tuple(corner)) + "is "
                        thought_process += "(" + str(corner[0]) + "-" + str(cloth_center[0]) + ")^2 + (" + str(corner[1]) + "-" + str(cloth_center[1]) + ")^2 = "
                        thought_process += str(int(np.linalg.norm(corner - cloth_center) ** 2)) + "\n"
                    distances = [np.linalg.norm(x - cloth_center) for x in cloth_corners]
                    distances = list(enumerate(distances))
                    distances = sorted(distances, key = lambda x : x[1], reverse=True)
                    indices = [index for index, _ in distances[:4]]
                    thought_process += "\nFrom this list above, we see that the four cloth corners that are the farthest from the center are:\n"
                    for index in indices:
                        thought_process += str(tuple(cloth_corners[index][0])) + "\n"

                    # Third step - Select the pick point as the corner that is the farthest from the center
                    thought_process += "From the above list of four corners, I will select the corner that is the farthest from the center as the picking point to achieve the maximum fold."
                    thought_process += "Hence the picking point here will be " + str(tuple(cloth_corners[indices[0]][0])) + ".\n"
                    
                    # Fourth step - Now select the point among the list of the other three points from the list above which is the closest to the point selected by us
                    thought_process += "Now we will select the point among the remaining three points of the above list which will be the closest to the picking point since we want to place it to the nearest edge"
                    test_pick_pixel = tuple(cloth_corners[indices[0]][0])
                    cloth_corners = cloth_corners[indices[1:]]
                    for (j, corner) in enumerate(cloth_corners):
                        corner = corner[0]
                        thought_process += str(j) + ": The distance for " + str(tuple(corner)) + " from " + str(test_pick_pixel) + " is "
                        thought_process += "(" + str(corner[0]) + "-" + str(test_pick_pixel[0]) + ")^2 + (" + str(corner[1]) + "-" + str(test_pick_pixel[1]) + ")^2 = "
                        thought_process += str(int(np.linalg.norm(corner - test_pick_pixel) ** 2)) + "\n"
                    distances = [np.linalg.norm(x - test_pick_pixel) for x in cloth_corners]
                    distances = list(enumerate(distances))
                    distances = sorted(distances, key = lambda x : x[1])
                    indices = [index for index, _ in distances]
                    test_place_pixel = tuple(cloth_corners[indices[0]][0])
                    thought_process += "\nFrom this list above, we see that the point that is the closest to the selected pick point is " + str(test_place_pixel) + "\n"

                    # Now saving these coorindates in the planning string as well
                    planning_string = "Planning:\n"
                    planning_string += "Pick Point = " + "(" + str(test_pick_pixel[0]) + ", " + str(test_pick_pixel[1]) + ")\n"
                    planning_string += "Place Point = " + "(" + str(test_place_pixel[0]) + ", " + str(test_place_pixel[1]) + ")\n"
                    step_json["assistant-response"] = planning_string + "\n" + thought_process
                    print(planning_string + "\n" + thought_process)
                    instructions_json[config_id][str(i)] = step_json

                print("The Pick and the place pixels", test_pick_pixel, test_place_pixel)

                pick_pixels.append(test_pick_pixel)
                place_pixels.append(test_place_pixel)

                # convert the pixel cords into world cords
                test_pick_pos = get_world_coord_from_pixel(test_pick_pixel, depth, camera_params)
                test_place_pos = get_world_coord_from_pixel(test_place_pixel, depth, camera_params)

                # pick & place
                env.pick_and_place(test_pick_pos.copy(), test_place_pos.copy())

                # render & update frames & save
                rgb, depth = env.render_image()
                depth_save = depth.copy() * 255
                depth_save = depth_save.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth_save)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                rgbs.append(rgb)

        elif args.task == "DoubleStraight":
            for i in range(4):
                # Getting a json object corresponding to each step
                step_json = {}

                # Detecting corners for the current cloth configuration
                image_path = os.path.join(save_folder_depth, str(i) + ".png")
                cloth_corners = find_corners(image_path)

                # Getting the user prompt and saving it in the json file
                user_prompt = get_user_prompt(cloth_corners, cloth_center, False, "", args.task, i)
                step_json["user-prompt"] = user_prompt
                step_json["system-prompt"] = system_prompt

                ### Now getting the expected assistant's response by performing logical operations on top of this
                thought_process = "Thought Process:\n"

                if len(cloth_corners) == 2:
                    thought_process += "The provided method of folding indicates that we have to select one corner of the cloth and fold it towards the edge, to align with the bottom edge.\n"
                    thought_process += "Now since there are only two corners provided to us, one of them will be the pick point and the other one will be the place point.\n\n"
                    pick_idx = 0
                    place_idx = 1
                    thought_process += "By picking the cloth at " + str(tuple(cloth_corners[pick_idx][0])) + " and placing it at the corner at " + str(tuple(cloth_corners[place_idx][0])) + " the resulting fold aligns with the instructions provided."
                else:
                    # First step - Just print generic stuff into it that could be used later. State that we need to select the top three farthest corners
                    thought_process += "The provided method of folding indicates that the cloth should be folded by picking a point on the top edge of the cloth and folding it downwards to align with the edge.\n\n"

                    thought_process += "There is a list of corners provided to me and I have to identify the point to be picked among these corners. To do so, I will first get the three corners that are the farthest from the center. The picking point and placing point will be one of these.\n\n"
                    thought_process += "I will use the square of the Euclidean distance between each corner and the center, " + str(cloth_center) +  ", and select the corners with the maximum distance.\n"

                    # Second step - Now, go through all the cloth corners and get the corner that is the farthest from the center
                    for (j, corner) in enumerate(cloth_corners):
                        corner = corner[0]
                        thought_process += str(j) + ": The distance for " + str(tuple(corner)) + "is "
                        thought_process += "(" + str(corner[0]) + "-" + str(cloth_center[0]) + ")^2 + (" + str(corner[1]) + "-" + str(cloth_center[1]) + ")^2 = "
                        thought_process += str(int(np.linalg.norm(corner - cloth_center) ** 2)) + "\n"
                    distances = [np.linalg.norm(x - cloth_center) for x in cloth_corners]
                    distances = list(enumerate(distances))
                    distances = sorted(distances, key = lambda x : x[1], reverse=True)
                    indices = [index for index, _ in distances[:3]]
                    thought_process += "\nFrom this list above, we see that the three cloth corners that are the farthest from the center are:\n"
                    corner_id = 0
                    for index in indices:
                        thought_process += str(corner_id) + ": " + str(tuple(cloth_corners[index][0])) + "\n"
                        corner_id += 1

                    cloth_corners = cloth_corners[indices]
                    print(cloth_corners)
                    # Third step - Now get the three possible pairs of combinations and get the pair with max distance
                    if i != 3:
                        thought_process += "\nTo decide the pick and place point, I will compute the distances across each pair of points among these three points above and get the pair with the lowest distance. I will again use the square of Euclidean distance here.\n"
                    else:
                        thought_process += "\nTo decide the pick and place point, I will compute the distances across each pair of points among these three points above and get the pair with the second-highest distance. I will again use the square of Euclidean distance here.\n"
                    thought_process += str(0) + ": The distance between the first and the second point is "
                    thought_process += "(" + str(cloth_corners[0][0][0]) + "-" + str(cloth_corners[1][0][0]) + ")^2 + (" + str(cloth_corners[0][0][1]) + "-" + str(cloth_corners[1][0][1]) + ")^2 = "
                    thought_process += str(int(np.linalg.norm(cloth_corners[0][0] - cloth_corners[1][0]) ** 2)) + "\n"
                    dist_1 = np.linalg.norm(cloth_corners[0][0] - cloth_corners[1][0])

                    thought_process += str(1) + ": The distance between the second and the third point is "
                    thought_process += "(" + str(cloth_corners[2][0][0]) + "-" + str(cloth_corners[1][0][0]) + ")^2 + (" + str(cloth_corners[2][0][1]) + "-" + str(cloth_corners[1][0][1]) + ")^2 = "
                    thought_process += str(int(np.linalg.norm(cloth_corners[2][0] - cloth_corners[1][0]) ** 2)) + "\n"
                    dist_2 = np.linalg.norm(cloth_corners[2][0] - cloth_corners[1][0])

                    thought_process += str(2) + ": The distance between the first and the third point is "
                    thought_process += "(" + str(cloth_corners[0][0][0]) + "-" + str(cloth_corners[2][0][0]) + ")^2 + (" + str(cloth_corners[0][0][1]) + "-" + str(cloth_corners[2][0][1]) + ")^2 = "
                    thought_process += str(int(np.linalg.norm(cloth_corners[0][0] - cloth_corners[2][0]) ** 2)) + "\n"
                    dist_3 = np.linalg.norm(cloth_corners[0][0] - cloth_corners[2][0])

                    pick_idx, place_idx = -1, -1
                    if dist_1 < dist_2 and dist_1 < dist_3:
                        if i != 3:
                            pick_idx, place_idx = 0, 1
                        else:
                            pick_idx, place_idx = (1, 2) if dist_2 < dist_3 else (0, 2)
                    elif dist_2 < dist_1 and dist_2 < dist_3:
                        if i != 3:
                            pick_idx, place_idx = 1, 2
                        else:
                            pick_idx, place_idx = (0, 1) if dist_1 < dist_3 else (0, 2)
                    else:
                        if i != 3:
                            pick_idx, place_idx = 0, 2
                        else:
                            pick_idx, place_idx = (1, 2) if dist_2 < dist_1 else (0, 1)

                    if i != 3:
                        thought_process += "\nHere we see that the points that are the closest are the point " + str(pick_idx) + " and the point " + str(place_idx) + "."
                        if i == 2:
                            pick_point = (cloth_corners[pick_idx][0])
                            place_point = (cloth_corners[place_idx][0])
                            if np.linalg.norm(pick_point - cloth_center) > np.linalg.norm(place_point - cloth_center):
                                pick_idx, place_idx = place_idx, pick_idx

                            thought_process += "\nThe distance of the point " + str(pick_idx) + " from the center is "
                            thought_process += "(" + str(pick_point[0]) + "-" + str(cloth_center[0]) + ")^2 + (" + str(pick_point[1]) + "-" + str(cloth_center[1]) + ")^2 = "
                            thought_process += str(int(np.linalg.norm(pick_point - cloth_center) ** 2)) + "\n"
                            thought_process += "The distance of the point " + str(place_idx) + " from the center is "
                            thought_process += "(" + str(place_point[0]) + "-" + str(cloth_center[0]) + ")^2 + (" + str(place_point[1]) + "-" + str(cloth_center[1]) + ")^2 = "
                            thought_process += str(int(np.linalg.norm(place_point - cloth_center) ** 2)) + "\n"

                            thought_process += "We will choose the point " + str(pick_idx) + " as the picking point as it is the closest to the center.\n"
                    else:
                        thought_process += "\nHere we see that the points that are the second-farthest are the point " + str(pick_idx) + " and the point " + str(place_idx) + "."
                    thought_process += "\nThus, by selecting these points as the pick and the place points, we achieve a straight fold that runs downward parallel to the edge.\n"
                    thought_process += "Note that here I did not choose the points that are the farthest from each other as it would result in a diagonal fold otherwise and it is not what we want here.\n"
                    thought_process += "By picking the cloth at " + str(tuple(cloth_corners[pick_idx][0])) + " and placing it at the corner at " + str(tuple(cloth_corners[place_idx][0])) + " the resulting fold aligns with the instructions provided."

                test_pick_pixel = np.array(cloth_corners[pick_idx][0])
                test_place_pixel = np.array(cloth_corners[place_idx][0])

                # Now saving these coorindates in the planning string as well
                planning_string = "Planning:\n"
                planning_string += "Pick Point = " + "(" + str(test_pick_pixel[0]) + ", " + str(test_pick_pixel[1]) + ")\n"
                planning_string += "Place Point = " + "(" + str(test_place_pixel[0]) + ", " + str(test_place_pixel[1]) + ")\n"
                step_json["assistant-response"] = planning_string + "\n" + thought_process
                print(planning_string + "\n" + thought_process)
                instructions_json[config_id][str(i)] = step_json

                test_pick_pixel = tuple(test_pick_pixel)
                test_place_pixel = tuple(test_place_pixel)
                print("The Pick and the place pixels", test_pick_pixel, test_place_pixel)

                pick_pixels.append(test_pick_pixel)
                place_pixels.append(test_place_pixel)

                # convert the pixel cords into world cords
                test_pick_pos = get_world_coord_from_pixel(test_pick_pixel, depth, camera_params)
                test_place_pos = get_world_coord_from_pixel(test_place_pixel, depth, camera_params)

                # pick & place
                env.pick_and_place(test_pick_pos.copy(), test_place_pos.copy())

                # render & update frames & save
                rgb, depth = env.render_image()
                depth_save = depth.copy() * 255
                depth_save = depth_save.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth_save)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                rgbs.append(rgb)

        particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]

        with open(os.path.join(save_folder, "info.pkl"), "wb+") as f:
            data = {"pick": pick_pixels, "place": place_pixels, "pos": particle_pos}
            pickle.dump(data, f)

        # action viz
        save_folder_viz = os.path.join(save_folder, "rgbviz")
        os.makedirs(save_folder_viz, exist_ok=True)

        num_actions = len(pick_pixels)

        for i in range(num_actions + 1):
            if i < num_actions:
                img = action_viz(rgbs[i], pick_pixels[i], place_pixels[i])
            else:
                img = rgbs[i]
            imageio.imwrite(os.path.join(save_folder_viz, str(i) + ".png"), img)

    # Saving the final dictionary for the current config
    with open(os.path.join(parent_directory, "utils", "gpt-demonstrations", args.task, "demonstrations.json"), "w") as f:
        json.dump(instructions_json, f, indent=4)

if __name__ == "__main__":
    main()
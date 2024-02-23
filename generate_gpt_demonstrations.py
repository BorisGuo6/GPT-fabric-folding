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
from utils.gpt_utils import system_prompt, get_user_prompt, analyze_images_gpt

def main():
    parser = argparse.ArgumentParser(description="Generate Demonstrations")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument("--img_size", type=int, default=224, help="Size of rendered image")
    parser.add_argument("--cached", type=str, help="Cached filename")
    args = parser.parse_args()

    # env settings
    cached_path = os.path.join("cached configs", args.cached + ".pkl")
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # demonstrator settings
    demonstrator = Demonstrator[args.task]()

    # save settings
    save_path = os.path.join("data", "gpt-demonstrations", args.task, args.cached, args.img_size)
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
            pick_idxs = demonstrator.pick_idxs
            for i, pick_idx in enumerate(pick_idxs):
                curr_corners = env.get_corners()
                pick_pos, place_pos = demonstrator.get_action(curr_corners, pick_idx)
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())
                rgb, depth = env.render_image()

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                rgbs.append(rgb)

        elif args.task == "AllCornersInward":
            for i in range(4):
                # Getting a json object corresponding to each step
                step_json = {}

                # Detecting corners for the current cloth configuration
                image_path = os.path.join(save_folder_depth, str(i) + ".png")
                cloth_corners = find_corners(image_path)

                # TODO - Get an instruction randomly from the saved list of instructions
                instruction = ""

                # Getting the user prompt and saving it in the json file
                user_prompt = get_user_prompt(cloth_corners, cloth_center, True, instruction, args.task)
                step_json["user-prompt"] = user_prompt
                step_json["system-prompt"] = system_prompt

                ### Now getting the expected assistant's response by performing logical operations on top of this
                thought_process = "Thought Process:\n"

                # First step - Just print generic stuff into it that could be used later. State that we need to selected the farthest corner
                thought_process += "The provided method of folding indicates that the cloth should be folded by bringing a corner to the center of the cloth.\n\n"
                thought_process += "There is a list of corners provided to me and I have to identify the point to be picked among these corners. To do so, I will select the corner point that is the farthest from the given cloth center in order to achieve the maximum fold.\n\n"
                thought_process += "I will use the square of the Euclidean distance between each corner and the center, " + str(cloth_center) +  ", and select the corner with the maximum distance.\n"

                # Second step - Now, go through all the cloth corners and get the corner that is the farthest from the center
                for (i, corner) in enumerate(cloth_corners):
                    thought_process += str(i) + ": The distance for " + str(corner) + "is "
                    thought_process += "(" + str(corner[0]) + "-" + str(cloth_center[0]) + ")^2 + (" + str(corner[1]) + "-" + str(cloth_center[1]) + ")^2 = "
                    thought_process += str(np.linalg.norm(corner - cloth_center) ** 2) + "\n"
                distances = [np.linalg.norm(x - cloth_center) for x in cloth_corners]
                index = np.argmax(distances)
                thought_process += "From this list above, we see that the cloth corner that is the farthest from the center is " + str(cloth_corners[index]) + "\n\n"

                # Third step - Now select the above point as the pick point and the center as the place point and save it to the response
                thought_process += "By picking the cloth at " + str(cloth_corners[index]) + " and placing it at the center at " + str(cloth_center) + " the resulting fold aligns with the instructions provided.\n"
                thought_process += "The pick and place pixels " + str(cloth_corners[index]) + " " + str(cloth_center)

                # Now saving these coorindates in the planning string as well
                planning_string = "Planning:\n"
                planning_string += "Pick Point = " + "(" + str(cloth_corners[index][0]) + ", " + str(cloth_corners[index][1]) + ")\n"
                planning_string += "Place Point = " + "(" + str(cloth_center[0]) + ", " + str(cloth_center[1]) + ")\n"
                step_json["assistant-response"] = planning_string + "\n" + thought_process
                instructions_json[config_id][str(i)] = step_json

                # Now performing the steps in simulation fr!
                pick_pos = get_world_coord_from_pixel(cloth_corners[index], depth, camera_params)
                place_pos = get_world_coord_from_pixel(cloth_center, depth, camera_params)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())
                rgb, depth = env.render_image()

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                rgbs.append(rgb)

        elif args.task == "CornersEdgesInward":
            center = env.get_center()
            for (i, pickplace_idx) in enumerate(demonstrator.pickplace_idxs):
                curr_corners = env.get_corners()
                edge_middles = env.get_edge_middles()
                pick_pos, place_pos = demonstrator.get_action(curr_corners, edge_middles, center, pickplace_idx)
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())
                rgb, depth = env.render_image()

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                rgbs.append(rgb)

        elif args.task == "DoubleStraight":
            pickplace_idxs = demonstrator.pickplace_idxs
            for (i, pickplace_idx) in enumerate(pickplace_idxs):
                curr_corners = env.get_corners()
                edge_middles = env.get_edge_middles()
                pick_pos, place_pos = demonstrator.get_action(curr_corners, edge_middles, pickplace_idx)
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())
                rgb, depth = env.render_image()

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
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
        # TODO

if __name__ == "__main__":
    main()
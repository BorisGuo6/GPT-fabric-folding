import argparse
import sys
import json
from datetime import date, timedelta
import time
import numpy as np
from softgym.envs.foldenv import FoldEnv
from utils.visual import get_world_coord_from_pixel, action_viz, save_video
import pyflex
import os
import pickle
from tqdm import tqdm
import imageio
from utils.gpt_utils import system_prompt, get_user_prompt, parse_output, analyze_images_gpt, gpt_v_demonstrations
from openai import OpenAI
from slurm_utils import find_corners, find_pixel_center_of_cloth, get_mean_particle_distance_error

from openai import OpenAI
# TODO: Use your own API key for performing the experiments
client = OpenAI(api_key="api_key")

def get_mask(depth):
    mask = depth.copy()
    mask[mask > 0.646] = 0
    mask[mask != 0] = 1
    return mask

def preprocess(depth):
    mask = get_mask(depth)
    depth = depth * mask
    return depth

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-Fabric for cloth folding tasks")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument("--img_size", type=int, default=128, help="Size of rendered image")
    parser.add_argument("--gpt_model", type=str, default="gpt-4-1106-preview", help="The GPT model that we wanna use for performing the folds")
    parser.add_argument("--cached", type=str, help="Cached filename")
    parser.add_argument('--save_video_dir', type=str, default='./videos/', help='Path to the saved video')
    parser.add_argument('--save_vid', type=bool, default=False, help='Decide whether to save video or not')
    parser.add_argument('--user_points', type=str, default="llm", help='Choose either user or llm')
    parser.add_argument('--total_runs', type=int, default=3, help='Total number of experiments that we wish to run for our system')
    parser.add_argument('--eval_type', type=str, default='in-context', help='Choose one of [zero-shot | in-context | fine-tuned] for GPT-Fabric')
    args = parser.parse_args()

    # IMPORTANT - This model will get deprecated on Dec 6, 2024. Kindly use any newer OpenAI models with vision reasoning abilities
    gpt_vision_model = "gpt-4-vision-preview"

    # task
    task = args.task
    if task == "CornersEdgesInward":
        steps = 4
    elif task == "AllCornersInward":
        steps = 4
    elif task == "DoubleStraight":
        steps = 3
    elif task == "DoubleTriangle":
        steps = 2

    # env settings
    cached_path = os.path.join("cached configs", args.cached + ".pkl")
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # The date when the experiment was run
    date_today = date.today()
    obtained_scores = np.zeros((args.total_runs, env.num_configs))
    time_array = np.zeros(env.num_configs)

    for run in tqdm(range(args.total_runs)):
        # Writing things to the specified log file
        output_file_path = os.path.join("logs", args.task, args.cached, str(date_today))
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        output_file = os.path.join(output_file_path, str(run) + ".log")
        sys.stdout = open(output_file, 'w', buffering=1)

        for config_id in tqdm(range(env.num_configs)):
            rgb_save_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "rgb")
            depth_save_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth")
            if not os.path.exists(rgb_save_path):
                os.makedirs(rgb_save_path)
            if not os.path.exists(depth_save_path):
                os.makedirs(depth_save_path)

            # record action's pixel info
            test_pick_pixels = []
            test_place_pixels = []
            rgbs = []

            # env settings
            env.reset(config_id=config_id)
            camera_params = env.camera_params

            # initial state
            rgb, depth = env.render_image()
            depth_save = depth.copy() * 255
            depth_save = depth_save.astype(np.uint8)
            imageio.imwrite(os.path.join(depth_save_path, "0.png"), depth_save)
            imageio.imwrite(os.path.join(rgb_save_path, "0.png"), rgb)
            rgbs.append(rgb)
            
            # Get the position of the center of the cloth in the initial configuration to pivot the required folds
            image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", "0.png")
            cloth_center = find_pixel_center_of_cloth(image_path)

            steps_time = np.zeros(steps)
            for i in range(steps):
                print("------------------------------------------------------")
                print("Currently in {} step of {} config in {} run".format(i, config_id, run))

                start_time = time.time()
                # get action based on the input asked to the user
                if args.user_points == "user":
                    # Detecting corners for the current cloth configuration
                    image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", str(i) + ".png")
                    cloth_corners = find_corners(image_path, False)

                    # Printing the detected cloth corners
                    print(cloth_corners)

                    pick_str = input("Enter the pick pixel in the form [x,y]: ")
                    test_pick_pixel = np.array(tuple(map(float, pick_str.strip("[]").split(','))))

                    place_str = input("Enter the place pixel in the form [x,y]: ")
                    test_place_pixel = np.array(tuple(map(float, place_str.strip("[]").split(','))))
                
                # get action based on what our LLM API integration predicts 
                elif args.user_points == "llm":
                    # Detecting corners for the current cloth configuration
                    image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", str(i) + ".png")
                    cloth_corners = find_corners(image_path)
                    
                    # Getting the template folding instruction images from the demonstrations
                    demo_root_path = os.path.join("data", "demo", args.task, "rgbviz")
                    start_image = os.path.join(demo_root_path, str(i) + ".png")
                    last_image = os.path.join(demo_root_path, str(i+1) + ".png")

                    # Generating the instruction by analyzing the images
                    instruction = analyze_images_gpt([start_image, last_image], args.task, i, args.eval_type, gpt_vision_model)

                    no_output = True
                    while no_output:
                        # getting the system and user prompts for our given request
                        user_prompt = get_user_prompt(cloth_corners, cloth_center, True, instruction, args.task, None)
                        print(user_prompt)

                        # Getting the demonstrations for in-context learning
                        indices = gpt_v_demonstrations[args.eval_type][args.task]["gpt-demonstrations"]
                        demonstration_dictionary_list = []

                        # This array will be empty for zero-shot evaluation of GPT-Fabric
                        if len(indices) != 0:
                            gpt_demonstrations_path = os.path.join("utils", "gpt-demonstrations", args.task, "demonstrations.json")
                            with open(gpt_demonstrations_path, 'r') as f:
                                gpt_demonstrations = json.load(f)
                            for index in indices:
                                step_dictionary = gpt_demonstrations[str(index)][str(i + 1)]
                                user_prompt_dictionary = {
                                    "role": "user",
                                    "content": step_dictionary["user-prompt"]
                                }
                                assistant_response_dictionary = {
                                    "role": "assistant",
                                    "content": step_dictionary["assistant-response"]
                                }
                                demonstration_dictionary_list += [user_prompt_dictionary, assistant_response_dictionary]

                        response = client.chat.completions.create(
                            model=args.gpt_model,
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
                        if test_pick_pixel.all() != None and test_place_pixel.all() != None:
                            no_output = False

                        # Printing the response given by the LLM in the log file generated
                        print(response.choices[0].message.content)
                
                else:
                    print("Falls back when the action is neither selected by the user or predicted by LLM")
                    exit(0)

                # The time taken by the LLM to generate this particular step
                steps_time[i] = time.time() - start_time
                
                # Appending the chosen pickels to the list of the pick and place pixels
                test_pick_pixel = np.array([min(args.img_size - 1, test_pick_pixel[0]), min(args.img_size - 1, test_pick_pixel[1])])
                test_place_pixel = np.array([min(args.img_size - 1, test_place_pixel[0]), min(args.img_size - 1, test_place_pixel[1])])
                test_pick_pixels.append(test_pick_pixel)
                test_place_pixels.append(test_place_pixel)

                # Printing the pixels chosen/computed
                print("The Pick and the place pixels", test_pick_pixel, test_place_pixel)

                # convert the pixel cords into world cords
                test_pick_pos = get_world_coord_from_pixel(test_pick_pixel, depth, camera_params)
                test_place_pos = get_world_coord_from_pixel(test_place_pixel, depth, camera_params)

                # pick & place
                env.pick_and_place(test_pick_pos.copy(), test_place_pos.copy())

                # render & update frames & save
                rgb, depth = env.render_image()
                depth_save = depth.copy() * 255
                depth_save = depth_save.astype(np.uint8)
                imageio.imwrite(os.path.join(depth_save_path, str(i + 1) + ".png"), depth_save)
                imageio.imwrite(os.path.join(rgb_save_path, str(i + 1) + ".png"), rgb)
                rgbs.append(rgb)

            # Saving the total time taken for this particular configuration
            time_array[config_id] = np.mean(steps_time)

            # Saving the final fold configuration in a pickle file
            particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
            with open(os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "info.pkl"), "wb+") as f:
                data = {"pick": test_pick_pixels, "place": test_place_pixels, "pos": particle_pos}
                pickle.dump(data, f)

            # action viz
            save_folder = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "rgbviz")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for i in range(steps + 1):
                if i < steps:
                    img = action_viz(rgbs[i], test_pick_pixels[i], test_place_pixels[i])
                else:
                    img = rgbs[i]
                imageio.imwrite(os.path.join(save_folder, str(i) + ".png"), img)

            # Save a video from the list of the image arrays
            if args.save_vid:
                save_vid_path = os.path.join(args.save_video_dir, args.task, args.cached, str(date_today), str(run))
                if not os.path.exists(save_vid_path):
                    os.makedirs(save_vid_path)
                save_video(env.rgb_array, os.path.join(save_vid_path, str(config_id)))
            env.rgb_array = []

            # Getting the score corresponding to the current run and the current config Id
            eval_dir = os.path.join("eval result", args.task, args.cached, str(date_today), str(run))
            expert_dir = os.path.join("data", "demonstrations", args.task, args.cached)
            score = get_mean_particle_distance_error(eval_dir, expert_dir, cached_path, args.task, config_id)
            obtained_scores[run,config_id] = score[0]

    # Saving the matrix corresponding to the obtained scores
    matrix_save_folder = os.path.join("position errors", args.task, args.cached)
    os.makedirs(matrix_save_folder, exist_ok=True)
    matrix_save_path = os.path.join(matrix_save_folder, str(date_today) + ".npy")
    np.save(matrix_save_path, obtained_scores)

    # Getting the analysis corresponding to the min and average obtained scores
    min_scores = np.zeros(env.num_configs)
    average_scores = np.zeros(env.num_configs)
    for config_id in range(env.num_configs):
        min_scores[config_id] = np.min(obtained_scores[:,config_id])
        average_scores[config_id] = np.mean(obtained_scores[:,config_id])
    
    # Printing the final values out
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # Note - We do not need to return the cherry picked scores anymore. So reverting that.
    # print("The mean and standard deviation for the best scores:", np.mean(min_scores), np.std(min_scores))
    print("The mean and standard deviation for the mean particle distance errors (mm):", np.mean(average_scores) * 1000, np.std(average_scores) * 1000)    

    # Printing the average time taken
    print("The average time taken by GPT-fabric for each configuration is:", np.mean(time_array))

if __name__ == "__main__":
    main()

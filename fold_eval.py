import argparse
import sys
import json
from datetime import date, timedelta
import time
import numpy as np
from softgym.envs.foldenv_bimanual import FoldEnv
from utils.visual import get_world_coord_from_pixel, action_viz, save_video
import pyflex
import os
import pickle
from tqdm import tqdm
import imageio
from utils.gpt_utils import system_prompt_bimanual, get_user_prompt, parse_output_bimanual, analyze_images_gpt, gpt_v_demonstrations
from openai import OpenAI
from slurm_utils import find_corners, find_pixel_center_of_cloth, get_mean_particle_distance_error

#deprectaed
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize

from openai import OpenAI
# TODO: Remove the Open AI api key before releasing the scripts in public
client = OpenAI(api_key="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")

def main():
    parser = argparse.ArgumentParser(description = "Bimanual Folding with GPT")
    parser.add_argument("--corners", action= "store_true", help= "Detect the corners of the given fabric")
    parser.add_argument("--folds", action= "store_true", help= "Performs the folding action on the given fabric")
    parser.add_argument("--cached", type=str, help="Cached filename")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument("--img_size", type=int, default=128, help="Size of rendered image")
    parser.add_argument('--save_video_dir', type=str, default='./videos/', help='Path to the saved video')
    parser.add_argument('--save_vid', action="store_true", help='Set if the video needs to be saved')
    parser.add_argument('--inputs', type=str, default="user", help='Defines whether the pick and place points are determined by the user of llm')
    parser.add_argument('--gpt_model', type=str, default="gpt-4", help='The version of gpt to be used')
    args = parser.parse_args()

    corners = args.corners

    config_id = 0
    run = 0

    if corners:
        print("slurm")

    # env settings
    cached_path = os.path.join("cached configs", args.cached + ".pkl")

    #deprecated
    '''

    env_kwargs = env_arg_dict['FoldEnv']
    #env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = True
    #env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    #env_kwargs['headless'] = args.headless

    env  = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    
    '''
    
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # The date when the experiment was run
    date_today = date.today()
    #obtained_scores = np.zeros((args.total_runs, env.num_configs))
    #time_array = np.zeros(env.num_configs)

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

    image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", "0.png")
    cloth_center = find_pixel_center_of_cloth(image_path)

    print(cloth_center)

    #image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", str(i) + ".png")
    cloth_corners = find_corners(image_path, False)

    # Printing the detected cloth corners
    print(cloth_corners)


    #Manually input the pick and place points from the user
 
    if args.inputs== "user":
        pick_str_1 = input("Enter the pick pixel in the form [x,y]: ")
        test_pick_pixel_1 = np.array(tuple(map(float, pick_str_1.strip("[]").split(','))))

        place_str_1 = input("Enter the place pixel in the form [x,y]: ")
        test_place_pixel_1 = np.array(tuple(map(float, place_str_1.strip("[]").split(','))))

        pick_str_2 = input("Enter the pick pixel in the form [x,y]: ")
        test_pick_pixel_2 = np.array(tuple(map(float, pick_str_2.strip("[]").split(','))))

        place_str_2 = input("Enter the place pixel in the form [x,y]: ")
        test_place_pixel_2 = np.array(tuple(map(float, place_str_2.strip("[]").split(','))))
        
    elif args.inputs == "llm":
        instructions_dict = {'DoubleStraight': "I want you to implement a double horizontal straight method of folding. Pick up the cloth along one of its top edge and fold it such that the picked up edge coincides with the opposite edge that is the bottom edge. The fold should be a horizontal fold",
                             'CornersInward': "I want you to implement a corner inward fold. In this case all the four corners of the the cloth need to be folded inwards. For the first step choose one of the four corners and fold it towards the center such that the chosen corner coincides with the center"}
        instruction = instructions_dict[args.task]
        user_prompt = get_user_prompt(cloth_corners, cloth_center, True, instruction, args.task, None)
        demonstration_dictionary_list = []
        response = client.chat.completions.create(
                            model=args.gpt_model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": system_prompt_bimanual
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
        print(response)
        print("\n \n")
        print(response.choices[0].message.content)
        test_pick_pixel_1, test_place_pixel_1, test_pick_pixel_2, test_place_pixel_2 = parse_output_bimanual(response.choices[0].message.content)
        #print(pi1, pl1, pi2, pl2)
    else:
        print("Your argument is invalid")

    # Appending the chosen pickels to the list of the pick and place pixels
    test_pick_pixel_1 = np.array([min(args.img_size - 1, test_pick_pixel_1[0]), min(args.img_size - 1, test_pick_pixel_1[1])])
    test_place_pixel_1 = np.array([min(args.img_size - 1, test_place_pixel_1[0]), min(args.img_size - 1, test_place_pixel_1[1])])

    ###################################
    #Add the condition such that the distance between the two pickers is greater than the radius of the picker
    ###################################

    test_pick_pixel_2 = np.array([min(args.img_size - 1, test_pick_pixel_2[0]), min(args.img_size - 1, test_pick_pixel_2[1])])
    test_place_pixel_2 = np.array([min(args.img_size - 1, test_place_pixel_2[0]), min(args.img_size - 1, test_place_pixel_2[1])])

    test_pick_pixels.append(test_pick_pixel_1)
    test_place_pixels.append(test_place_pixel_1)

    test_pick_pos_1 = get_world_coord_from_pixel(test_pick_pixel_1, depth, camera_params)
    test_place_pos_1= get_world_coord_from_pixel(test_place_pixel_1, depth, camera_params)

    
    test_pick_pos_2 = get_world_coord_from_pixel(test_pick_pixel_2, depth, camera_params)
    test_place_pos_2= get_world_coord_from_pixel(test_place_pixel_2, depth, camera_params)
    
    
    
    test_pick_pos = np.vstack((test_pick_pos_1, test_pick_pos_2))
    test_place_pos = np.vstack((test_place_pos_1, test_place_pos_2))
    
    #print(np.shape(test_pick_pos))
    # pick & place
    env.pick_and_place(test_pick_pos.copy(), test_place_pos.copy())

    rgb, depth = env.render_image()
    depth_save = depth.copy() * 255
    depth_save = depth_save.astype(np.uint8)
    imageio.imwrite(os.path.join(depth_save_path, str(1) + ".png"), depth_save)
    imageio.imwrite(os.path.join(rgb_save_path, str(1) + ".png"), rgb)
    rgbs.append(rgb)

    run = 25_1
    if args.save_vid:
        save_vid_path = os.path.join(args.save_video_dir, args.task, args.cached, str(date_today), str(run))
        if not os.path.exists(save_vid_path):
                os.makedirs(save_vid_path)
        save_video(env.rgb_array, os.path.join(save_vid_path, str(config_id)))


if __name__ == "__main__":
    main()
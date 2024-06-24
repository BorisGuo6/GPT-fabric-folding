#This file uses the code generation abilities of GPT for a bimanual fold system

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
from utils.gpt_utils import system_prompt_bimanual, get_user_prompt, parse_output_bimanual, analyze_images_gpt, gpt_v_demonstrations, set_of_marks
from openai import OpenAI
from slurm_utils import find_corners, find_pixel_center_of_cloth, get_mean_particle_distance_error, append_pixels_to_list_bimanual
from cv2 import imwrite, imread
from prompts_2 import PROMPT_1, PROMPT_2, PROMPT_3, PROMPT_4, PROMPT_5, PROMPT_6, MAIN_PROMPT, ERROR_CORRECTION_PROMPT
from gpt import generate_code_from_gpt
import traceback
from contextlib import redirect_stdout
from gpt import generate_code_from_gpt

#deprectaed
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize

from openai import OpenAI
# TODO: Remove the Open AI api key before releasing the scripts in public
client = OpenAI(api_key="sk-proj-gpjnKOl4bOwfXhGQQfbVT3BlbkFJUe0DAyVZTe1G6oKNubnD")
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Bimanual Folding with GPT")
    parser.add_argument("--cached", type=str, help="Cached filename")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument("--img_size", type=int, default=128, help="Size of rendered image")
    parser.add_argument('--save_video_dir', type=str, default='./videos/', help='Path to the saved video')
    parser.add_argument('--save_vid', action="store_true", help='Set if the video needs to be saved')
    parser.add_argument('--inputs', type=str, default="user", help='Defines whether the pick and place points are determined by the user of llm')
    parser.add_argument('--gpt_model', type=str, default="gpt-4", help='The version of gpt to be used')
    parser.add_argument('--eval_type', type=str, help="Mention whether [zero-shot | in-context] learning")
    parser.add_argument('--total_runs', type=int, help="Mention the total number of runs")
    args = parser.parse_args()


    cached_path = os.path.join("cached configs", args.cached + ".pkl")

    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # env settings
    cached_path = os.path.join("cached configs", args.cached + ".pkl")
    
    
    img_size = args.img_size

    
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # The date when the experiment was run
    date_today = date.today()
    obtained_scores = np.zeros((args.total_runs, env.num_configs))
    #time_array = np.zeros(env.num_configs)

    for run in tqdm(range(args.total_runs)):
        for config_id in tqdm(range(env.num_configs)):
            if config_id ==10:
                break

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
            pick_and_place = env.pick_and_place

            # initial state
            rgb, depth = env.render_image()
            depth_save = depth.copy() * 255
            depth_save = depth_save.astype(np.uint8)
            imageio.imwrite(os.path.join(depth_save_path, "0.png"), depth_save)
            imageio.imwrite(os.path.join(rgb_save_path, "0.png"), rgb)
            rgbs.append(rgb)
            
            #temporarily deprecated
            '''
            #image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", str(i) + ".png")
            cloth_corners_2 = find_corners("/home/rajeshgayathri2003/GPT-fabric-folding/goals/square_towel/single_step/depth/test_goal_9_depth.png", False).squeeze()
            '''
            
            
            #temporarily deprecated
            #set of marks
            '''
            im = imread("/home/rajeshgayathri2003/GPT-fabric-folding/goals/square_towel/start.png")
            image = set_of_marks(rgb, cloth_corners, regions= False)
            imwrite("/home/rajeshgayathri2003/GPT-fabric-folding/videos/CornersInward/square/2024-06-05/251/trial1.png", image)
            image_2 = set_of_marks(imread("/home/rajeshgayathri2003/GPT-fabric-folding/goals/square_towel/single_step/rgb/test_goal_9.png"), cloth_corners_2, cloth_center)
            imwrite("/home/rajeshgayathri2003/GPT-fabric-folding/videos/CornersInward/square/2024-06-05/251/trial1_2.png", image_2)
            '''
            #Manually input the pick and place points from the user
            
            steps = {"DoubleTriangle": 2,
                    "DoubleStraightBimanual": 2,
                    "CornersInward": 2}
        
            if args.inputs== "user":
                image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", "0.png")
                cloth_center = find_pixel_center_of_cloth(image_path)
                print(cloth_center)
                
                cloth_corners = find_corners(image_path, False)
                # Printing the detected cloth corners
                print(cloth_corners)

                pick_str_1 = input("Enter the pick pixel in the form [x,y]: ")
                test_pick_pixel_1 = np.array(tuple(map(float, pick_str_1.strip("[]").split(','))))

                place_str_1 = input("Enter the place pixel in the form [x,y]: ")
                test_place_pixel_1 = np.array(tuple(map(float, place_str_1.strip("[]").split(','))))

                pick_str_2 = input("Enter the pick pixel in the form [x,y]: ")
                test_pick_pixel_2 = np.array(tuple(map(float, pick_str_2.strip("[]").split(','))))

                place_str_2 = input("Enter the place pixel in the form [x,y]: ")
                test_place_pixel_2 = np.array(tuple(map(float, place_str_2.strip("[]").split(','))))
                
            elif args.inputs == "llm":
                
                PROMPTS = [PROMPT_1, PROMPT_2, PROMPT_3, PROMPT_4, PROMPT_5, PROMPT_6]
                step = steps[args.task]
                for i in range(0, step):
                    print("############Step{}#############".format(i))
                    demo_root_path = os.path.join("data", "demo", args.task, "rgbviz")
                    start_image = os.path.join(demo_root_path, str(i) + ".png")
                    last_image = os.path.join(demo_root_path, str(i+1) + ".png")
                    
                    # Generating the instruction by analyzing the images
                    instruction = analyze_images_gpt([start_image, last_image], args.task, i, args.eval_type)
                    print(instruction)

                    image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", str(i)+".png")
                    cloth_center = find_pixel_center_of_cloth(image_path)
                    print(cloth_center)
                
                    cloth_corners = find_corners(image_path, False)
                    # Printing the detected cloth corners
                    print(cloth_corners)
                    
                    # instructions_dict = {0:{'DoubleStraightBimanual': "I want you to implement a double straight method of folding. Pick up the cloth along one of its vertical edge and fold it such that the picked up edge coincides with the opposite edge. This generates the required fold",
                    #                     'CornersInward': "I want you to implement a corner inward fold. In this case all the four corners of the the cloth need to be folded inwards. For the first step choose two of the four corners and fold it towards the center such that the chosen corner coincides with the center"},
                    #                     1:{'DoubleStraightBimanual': "I want you to implement the second fold of the double straight method of folding. I want you to pick up one of the top edges of the cloth and fold it such that it coincides with its bottom edge. This is the required horizontal fold.",
                    #                     'CornersInward': "As a second step, choose the two corners that were not chosen earlier and fold these two towards the center"}
                    # }
                    # instruction = instructions_dict[i][args.task]
                    
                    
                    new_prompt = MAIN_PROMPT
                    messages = [{"role": "system", "content":new_prompt}]
                    # generate = False
                    # if generate:
                    for val, prompt in enumerate(PROMPTS):
                        if prompt == PROMPT_2:
                            prompt = prompt.replace("[INSERT TASK]", instruction)

                        content = generate_code_from_gpt(args.gpt_model, client, prompt, i, config_id, val, "user", messages)
                    
                        completed = False
                        new_prompt = ""
                        error = None

                        loc = locals()
                        error_count = 0
                                
                        while not(completed):
                            if error_count>=3:
                                messages = [{"role": "system", "content":new_prompt}]
                                content = generate_code_from_gpt(args.gpt_model, client, prompt, i, config_id, val, "user", messages)
                            flag = False
                            sys.stdout = sys.__stdout__
                            block_number = 0
                          
                            code_block = content.split("```python")

                            code = [block.split("```")[0] for block in code_block if len(block.split("```")) > 1]
                            code = code[0] if len(code)>0 else None
                            if code is None:
                                print(code_block)
                                break
                            try:       
                                exec(code, globals(), loc)
                                            
                            except Exception:
                                error_count+=1
                                if error_count>3:
                                    continue
                                error_message = traceback.format_exc()
                                print(error_message)
                                new_prompt+=ERROR_CORRECTION_PROMPT.replace("[INSERT BLOCK NUMBER]", str(block_number)).replace("[INSERT ERROR MESSAGE]", error_message)
                                new_prompt+="\n"
                                error = True
                                flag = True
                                content = generate_code_from_gpt(args.gpt_model, client, new_prompt, i, config_id, val, "user", messages)
                                continue
                                        
                            else:    
                                if not(flag):
                                    error = False
                                    completed = True
                                    rgb, depth = env.render_image()
                                    depth_save = depth.copy() * 255
                                    depth_save = depth_save.astype(np.uint8)
                                    imageio.imwrite(os.path.join(depth_save_path, str(i + 1) + ".png"), depth_save)
                                    imageio.imwrite(os.path.join(rgb_save_path, str(i + 1) + ".png"), rgb)
                                    rgbs.append(rgb)
                                    print("appended")
                                    #print(pick_point)
                                                    
                                    print("The value is", completed)        

                            print("done")
                                        
                                
                                            
                            
                                
                                    
                                    
                    # file = "log_{}_1.txt".format(i)
                    # print(file)
                    # f = open(file, 'r')
                    # content = f.read()
            
                    # code_block = content.split("```python")
                    # sys.stdout = sys.__stdout__
                    # block_number = 0
                    # loc = locals()
                    # for block in code_block:
                    #     if len(block.split("```")) > 1:
                    #         code = block.split("```")[0]
                    #         block_number+=1   
                                
                    #         exec(code, globals(), loc)
                                
                        
                    
                    
                    
                    # Appending the chosen pickels to the list of the pick and place pixels
                    # test_pick_pixel_1 = np.array([min(args.img_size - 1, pick_pos_1[0]), min(args.img_size - 1, pick_pos_1[1])])
                    # test_place_pixel_1 = np.array([min(args.img_size - 1, place_pos_1[0]), min(args.img_size - 1, place_pos_1[1])])

                    # ###################################
                    # #Add the condition such that the distance between the two pickers is greater than the radius of the picker
                    # ###################################

                    # test_pick_pixel_2 = np.array([min(args.img_size - 1, pick_pos_2[0]), min(args.img_size - 1, pick_pos_2[1])])
                    # test_place_pixel_2 = np.array([min(args.img_size - 1, place_pos_2[0]), min(args.img_size - 1, place_pos_2[1])])

                    # #check this part it might cause issues later
                    # test_pick_pixels.append(np.vstack((test_pick_pixel_1, test_pick_pixel_2)))
                    # test_place_pixels.append(np.vstack((test_place_pixel_1, test_place_pixel_2)))

                    # print(test_pick_pixel_1)
                    # print(type(test_pick_pixel_1))
                    
                    # test_pick_pos_1 = get_world_coord_from_pixel(test_pick_pixel_1, depth, camera_params)
                    # test_place_pos_1= get_world_coord_from_pixel(test_place_pixel_1, depth, camera_params)

                    
                    # test_pick_pos_2 = get_world_coord_from_pixel(test_pick_pixel_2, depth, camera_params)
                    # test_place_pos_2= get_world_coord_from_pixel(test_place_pixel_2, depth, camera_params)
                    
                    
                    
                    # test_pick_pos = np.vstack((test_pick_pos_1, test_pick_pos_2))
                    # test_place_pos = np.vstack((test_place_pos_1, test_place_pos_2))
                    
                    # #print(np.shape(test_pick_pos))
                    # # pick & place
                    # env.pick_and_place(test_pick_pos.copy(), test_place_pos.copy())

                    # # render & update frames & save
                    # rgb, depth = env.render_image()
                    # depth_save = depth.copy() * 255
                    # depth_save = depth_save.astype(np.uint8)
                    # imageio.imwrite(os.path.join(depth_save_path, str(i + 1) + ".png"), depth_save)
                    # imageio.imwrite(os.path.join(rgb_save_path, str(i + 1) + ".png"), rgb)
                    # rgbs.append(rgb)

                particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
                with open(os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "info.pkl"), "wb+") as f:
                    data = {"pick": test_pick_pixels, "place": test_place_pixels, "pos": particle_pos}
                    pickle.dump(data, f)

                save_folder = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "rgbviz")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                # for i in range(step + 1):
                #     if i < step:
                #         img = action_viz(rgbs[i], test_pick_pixels[i], test_place_pixels[i])
                #     else:
                #         img = rgbs[i]
                #     imageio.imwrite(os.path.join(save_folder, str(i) + ".png"), img)

                
                if args.save_vid:
                    save_vid_path = os.path.join(args.save_video_dir, args.task, args.cached, str(date_today), str(run))
                    if not os.path.exists(save_vid_path):
                            os.makedirs(save_vid_path)
                    save_video(env.rgb_array, os.path.join(save_vid_path, str(config_id)))
                            
                eval_dir = os.path.join("eval result", args.task, args.cached, str(date_today), str(run))
                expert_dir = os.path.join("data", "demonstrations", args.task, args.cached)
                score = get_mean_particle_distance_error(eval_dir, expert_dir, cached_path, args.task, config_id)
                obtained_scores[run,config_id] = score[0]
                        
                        #print(pi1, pl1, pi2, pl2)
            else:
                print("Your argument is invalid")


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
    print("The mean and standard deviation for the best scores:", np.mean(min_scores), np.std(min_scores))
    print("The mean and standard deviation for the average scores:", np.mean(average_scores), np.std(average_scores))

        


# if __name__ == "__main__":
#     main()
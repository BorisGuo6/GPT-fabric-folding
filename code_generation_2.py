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
from utils.gpt_utils import get_user_prompt, parse_output_bimanual, analyze_images_gpt, gpt_v_demonstrations, set_of_marks
from openai import OpenAI
from slurm_utils import find_corners, find_pixel_center_of_cloth, get_mean_particle_distance_error, append_pixels_to_list
from cv2 import imwrite, imread
from prompts import MAIN_PROMPT, ERROR_CORRECTION_PROMPT
from io import StringIO
from contextlib import redirect_stdout
import traceback

#deprectaed
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize


from openai import OpenAI
# TODO: Remove the Open AI api key before releasing the scripts in public
client = OpenAI(api_key="sk-proj-gpjnKOl4bOwfXhGQQfbVT3BlbkFJUe0DAyVZTe1G6oKNubnD")
#

def main():
    parser = argparse.ArgumentParser(description = "Code generating abilities of GPT")
    parser.add_argument("--cached", type=str, help="Cached filename")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument("--img_size", type=int, default=128, help="Size of rendered image")
    parser.add_argument('--save_video_dir', type=str, default='./videos/', help='Path to the saved video')
    parser.add_argument('--save_vid', action="store_true", help='Set if the video needs to be saved')
    parser.add_argument('--inputs', type=str, default="user", help='Defines whether the pick and place points are determined by the user of llm')
    parser.add_argument('--gpt_model', type=str, default="gpt-4", help='The version of gpt to be used')
    parser.add_argument('--eval_type', type=str, help="Mention whether [zero-shot | in-context] learning")
    args = parser.parse_args()

    cached_path = os.path.join("cached configs", args.cached + ".pkl")

    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)
    pick_and_place = env.pick_and_place
    
    #temporary
    run = 0
    config_id = 0
    
    
    #today's date
    date_today = date.today()
    
    #saving path
    rgb_save_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "rgb")
    depth_save_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth")
    if not os.path.exists(rgb_save_path):
        os.makedirs(rgb_save_path)
    if not os.path.exists(depth_save_path):
        os.makedirs(depth_save_path)

    #current_pos, _ = env.action_tool._get_pos()[0].squeeze()
    
    # env settings
    env.reset(config_id=config_id)
    camera_params = env.camera_params
    
    # record action's pixel info
    test_pick_pixels = []
    test_place_pixels = []
    rgbs = []

    # initial state
    rgb, depth = env.render_image()
    depth_save = depth.copy() * 255
    depth_save = depth_save.astype(np.uint8)
    imageio.imwrite(os.path.join(depth_save_path, "0.png"), depth_save)
    imageio.imwrite(os.path.join(rgb_save_path, "0.png"), rgb)
    rgbs.append(rgb)
    
    
    #instruction = "The double triangle fold involves two steps. In step one, the cloth is folded along its diagonal.This brings one of the top corners to one of the bottom corners on the opposite side resulting in a triangular shape. In step two, the triangular shaped cloth is folded along its longer side to get a smaller triangle."
    instructions = {0: "In step one, the cloth is folded along its diagonal.This brings one of the top corners to one of the bottom corners on the opposite side resulting in a triangular shape.",
                   1: "In step two, the triangular shaped cloth is folded along its longer side to get a smaller triangle. In this way one of the top corners will coincide with the bottom corner of the opposite side."

    }
    steps = 2
    count = 0
    #print(code_block)
    
    
    i = 0

    for step in range(steps):
        image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", str(count) + ".png")
        new_prompt = ""
        print("###############Step {}##################".format(step))
        completed = False
        error = None
        instruction = instructions[step]
        f = open("log.txt".format(step), 'r')
        content = f.read()
        
        code_block = content.split("```python")
        sys.stdout = sys.__stdout__
        block_number = 0
        
        for block in code_block:
            if len(block.split("```")) > 1:
                code = block.split("```")[0]
                block_number+=1   
                
                exec(code)
        count+=1
        
        rgb, depth = env.render_image()
        depth_save = depth.copy() * 255
        depth_save = depth_save.astype(np.uint8)
        imageio.imwrite(os.path.join(depth_save_path, str(i + 1) + ".png"), depth_save)
        imageio.imwrite(os.path.join(rgb_save_path, str(i + 1) + ".png"), rgb)
        rgbs.append(rgb)            
        
        
        '''
        while not completed:
            flag = False
            
            for block in code_block:
                
                if len(block.split("```")) > 1:
                    code = block.split("```")[0]
                    block_number+=1
                    
                    try:
                        exec(code)
                    except Exception:
                        
                        error_message = traceback.format_exc()
                        print(error_message)

                        new_prompt+=ERROR_CORRECTION_PROMPT.replace("[INSERT BLOCK NUMBER]", str(block_number)).replace("[INSERT ERROR MESSAGE]", error_message)
                        new_prompt+="\n"
                        error = True
                        flag = True
                        break
                    else:
                        if not(flag):
                            error = False   
                        
                else:
                    print("passc")

            print("done")
                
            if error:
                pass
            
                file = "/home/rajeshgayathri2003/GPT-fabric-folding/log.txt"
                
                sys.stdout = open(file, 'w')
                
            
            else:
                error = False
                if not(error) and not(flag):
                    completed = True
                    count+=1
                #print(pick_point)
                print("The value of count is", count)
                print("The value is", completed)
                
        '''

                
    run = 25_2
    if args.save_vid:
        save_vid_path = os.path.join(args.save_video_dir, args.task, args.cached, str(date_today), str(run))
        if not os.path.exists(save_vid_path):
            os.makedirs(save_vid_path)
        save_video(env.rgb_array, os.path.join(save_vid_path, str(config_id)))
            
    print("Done")

    
    #deprecated 
    '''
    for chunk in response:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="", file=file)
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason, file=file)
    '''

    


    

if __name__ == "__main__":
    main()

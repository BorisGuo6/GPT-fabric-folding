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
from utils.gpt_utils import get_user_prompt, parse_output_bimanual, analyze_images_gpt, gpt_v_demonstrations, set_of_marks
from openai import OpenAI
from slurm_utils import find_corners, find_pixel_center_of_cloth, get_mean_particle_distance_error
from cv2 import imwrite, imread
from prompts import MAIN_PROMPT

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

    #current_pos, _ = env.action_tool._get_pos()[0].squeeze()

    instruction = "In the double triangle fold, the cloth is folded along its diagonal in the first step. This brings one of the top corners to one of the bottom corners on the opposite side resulting in a triangular shape. In the second step, the traingle is again folded along its longest side. This again results in one of the top corners coinciding with the bottom corner on the opposite side."

    new_prompt = MAIN_PROMPT.replace("[INSERT TASK]", instruction)

    messages = []
    messages.append({
                        "role": "system",
                        "content": new_prompt
                    })
    response = client.chat.completions.create(
    
                                model=args.gpt_model,
                                messages=messages,
                                temperature=0,
                                max_tokens=769,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0
                            )


    new_output = ""
    file = "/home/rajeshgayathri2003/GPT-fabric-folding/log.txt"
    content = response.choices[0].message.content
    sys.stdout = open(file, 'w')
    print(content)

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

    messages.append({"role":"assistant", "content":new_output})


    

if __name__ == "__main__":
    main()

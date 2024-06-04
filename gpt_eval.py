import os
import time 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from omegaconf import OmegaConf
import imageio
from slurm_utils import find_corners, find_pixel_center_of_cloth, get_mean_particle_distance_error
from datetime import date, timedelta
import pyflex
from softgym.envs.bimanual_env import BimanualEnv
from softgym.envs.bimanual_tshirt import BimanualTshirtEnv

'''
class EnvRollout(object):
    def __init__(self, args):
        self.args = args
        

        if 'towel' in args.cloth_type:
            
            self.env = BimanualEnv(use_depth=True,
                    use_cached_states=False,
                    horizon=1,
                    action_repeat=1,
                    headless=args.headless,
                    shape='default' if 'square' in args.cloth_type else 'rect')
        elif args.cloth_type == 'tshirt':
            self.env = BimanualTshirtEnv(use_depth=True,
                    use_cached_states=False,
                    horizon=1,
                    action_repeat=1,
                    headless=args.headless)

    
    def load_image(self, args):
        cached_path = os.path.join("cached configs", args.cached + ".pkl")
        date_today = date.today()

        run = 0
        config_id = 0

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

        #camera_params = self.env.camera_params

        # initial state
        print(1)
        rgb, depth = self.env.get_rgbd(128)
        print(2)
        depth_save = depth.copy() * 255
        depth_save = depth_save.astype(np.uint8)
        imageio.imwrite(os.path.join(depth_save_path, "0.png"), depth_save)
        imageio.imwrite(os.path.join(rgb_save_path, "0.png"), rgb)
        rgbs.append(rgb)

        image_path = os.path.join("eval result", args.task, args.cached, str(date_today), str(run), str(config_id), "depth", "0.png")
        cloth_center = find_pixel_center_of_cloth(image_path)
        print(cloth_center)
'''
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloth_type', help='cloth type to load', default='square_towel', choices=['square_towel', 'rect_towel', 'tshirt'])
    parser.add_argument("--cached", type=str, help="Cached filename")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument('--headless', help='Run headless evaluation', action='store_true')
    '''
    parser.add_argument('--single_pick_thresh', help='min px distance to switch dual pick to single pick', default=30)
    parser.add_argument('--action_len_thresh', help='min px distance for an action', default=10)
    parser.add_argument('--goal_repeat', help='Number of times to repeat one goal', default=1)
    parser.add_argument('--seed', help='random seed', default=0)
    parser.add_argument('--headless', help='Run headless evaluation', action='store_true')
    parser.add_argument('--crumple_idx', help='index for crumpled initial configuration, set to -1 for no crumpling', type=int, default=-1)
    '''
    args = parser.parse_args()
    rgb, depth = pyflex.render()
    if 'towel' in args.cloth_type:
        pass
            
        env = BimanualEnv(use_depth=True,
                    use_cached_states=False,
                    horizon=1,
                    action_repeat=1,
                    headless=args.headless,
                    shape='default' if 'square' in args.cloth_type else 'rect')
    elif args.cloth_type == 'tshirt':
        pass
        env = BimanualTshirtEnv(use_depth=True,
                    use_cached_states=False,
                    horizon=1,
                    action_repeat=1,
                    headless=args.headless)
            
    
    cached_path = os.path.join("cached configs", args.cached + ".pkl")
    date_today = date.today()

    run = 0
    config_id = 0

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
    
    
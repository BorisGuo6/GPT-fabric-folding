import os.path as osp
import argparse
import numpy as np
import time 
import cv2


from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
from matplotlib import pyplot as plt

import imageio
from slurm_utils import find_corners, find_pixel_center_of_cloth, get_mean_particle_distance_error
from datetime import date, timedelta
import pyflex
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.envs.bimanual_env import BimanualEnv
from softgym.envs.bimanual_tshirt import BimanualTshirtEnv



def show_depth():
    # render rgb and depth
    img, depth = pyflex.render()
    img = img.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape((720, 720))[::-1]
    # get foreground mask
    rgb, depth = pyflex.render_cloth()
    depth = depth.reshape(720, 720)[::-1]
    # mask = mask[:, :, 3]
    # depth[mask == 0] = 0
    # show rgb and depth(masked)
    depth[depth > 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[1].imshow(depth)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    frames = [env.get_image(args.img_size, args.img_size)]
    '''print("hello", env.horizon)
    for i in range(env.horizon):
        action = env.action_space.sample()
        print(np.shape(action))
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        frames.extend(info['flex_env_recorded_frames'])
        if args.test_depth:
            show_depth()
    '''
    
    for i in range(env.horizon):
        curr_picker_pos=env.action_tool._get_pos()[0].squeeze()
        #print(curr_picker_pos.dtype)
        target_pos = np.array([[ 0.09, 0 , 0.12], [0, 0, 0]], dtype = np.float32)
        #target_pos_2 = np.array([[0.09, 0, 0.15], [ 0.15, 0 , 0]], dtype = np.float32)
        picker_translation = target_pos - curr_picker_pos
        action_lst = []
        
        picker_action=np.append(picker_translation,[[1],[1]], axis = 1).flatten()
        action_lst.append(picker_action)
        
        #curr_picker_pos_2=env.action_tool._get_pos()[0].squeeze()
        #picker_translation_2 = target_pos_2 - curr_picker_pos_2
        #picker_action_2=np.append(picker_translation_2,[[0],[0]], axis = 1).flatten()
        #action_lst.append(picker_action_2)
        ###print(picker_action)
        #print(np.shape(curr_picker_pos), np.shape(target_pos))
        action = env.action_space.sample()
        ###print((action))
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        for picker_action in action_lst:
            _, _, _, info = env.step(picker_action, record_continuous_video=True, img_size=args.img_size)
            frames.extend(info['flex_env_recorded_frames'])
        if args.test_depth:
            show_depth()
            
    
            
    env_b = BimanualEnv(use_depth=True,
                    use_cached_states=False,
                    horizon=1,
                    action_repeat=1,
                    headless= env_kwargs['headless'],
                    shape='default')
    
    #env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env_b.reset()
    
    rgb, depth = pyflex.render()

            
    if args.save_video_dir is not None:
        save_name = osp.join(args.save_video_dir, args.env_name + '_new2.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()

import os
import cv2
import numpy as np
import imageio
import glob
from PIL import Image
from moviepy.editor import ImageSequenceClip


def make_grid(array, nrow=1, padding=0, pad_value=120):
    """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    assert N % nrow == 0
    ncol = N // nrow
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(array[idx], [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]], constant_values=pad_value)
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(array[idx], [[padding if i == 0 else 0, padding], [0, padding], [0, 0]], constant_values=pad_value)
            row = np.hstack([row, cur_img])
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
    return grid_img


if __name__ == '__main__':
    N = 12
    H = W = 50
    X = np.random.randint(0, 255, size=N * H * W* 3).reshape([N, H, W, 3])
    grid_img = make_grid(X, nrow=3, padding=5)
    cv2.imshow('name', grid_img / 255.)
    cv2.waitKey()


def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def save_numpy_to_gif_matplotlib(array, filename):
    import subprocess

    def save_images_as_files(image_list, output_folder='temp_frames'):
        # Create a temporary directory to store individual frame images
        subprocess.call(['mkdir', output_folder])

        # Save each 2D array as an individual image file (e.g., PNG)
        for i, img_array in enumerate(image_list):
            imageio.imwrite(f'{output_folder}/frame_{i:04d}.png', img_array)

        return output_folder

    def create_video_from_images(image_folder, output_path, fps=24):
        # Use ffmpeg to create a video from the saved image files
        subprocess.call(['ffmpeg', '-framerate', str(fps), '-i', f'{image_folder}/frame_%04d.png', '{}.mp4'.format(filename)])

    # Replace 'image_list' with your actual list of 2D NumPy arrays
    image_list = array

    # Save the 2D arrays as individual image files
    image_folder = save_images_as_files(image_list)

    # Create the video from the saved image files
    create_video_from_images(image_folder, '{}.mp4'.format(filename), fps=24)

    # Remove the temporary directory and its contents
    subprocess.call(['rm', '-r', image_folder])
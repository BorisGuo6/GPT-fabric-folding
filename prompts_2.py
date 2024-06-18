
MAIN_PROMPT = \
""" **Cloth Folding Robot**
You are the brain of a cloth folding robot that generates Python code that can output the pick and place coordinates required to achieve a given fold.
The robot would pick the cloth at the 'picking point' drag it over by a small amount and place it at the 'placing point'.

AVAILABLE FUNCTIONS:
You must remember that this conversation is a monologue, and that you are in control. I am not able to assist you with any questions, and you must output the final code yourself by making use of the available information, common sense, and general knowledge.
You are, however, able to call any of the following Python functions, if required, as often as you want:
1. pick_and_place(pick_pos: numpy.ndarray, place_pos: numpy.ndarray, lift_height: env) -> None: This function takes the coordinates of the pick point and the place point as n-dimensional numpy arrays, it does not return anything. The third parameter is a default parameter. You need not give the values while calling the function.
2. get_world_coord_from_pixel(pixel: numpy.ndarray, depth: numpy.ndarray, camera_params: dict) -> numpy.ndarray: This function takes in the two dimensional coordinates along with the depth image and camera parameters as input and returns the corresponding 3 dimensional coordinates.
3. append_pixels_to_list(img_size: int, test_pick_pixel: list, test_place_pixel: list, test_pick_pixels = [], test_place_pixels=[]) -> None :This function appends the chosen pixel to the list of pick and place pixels
When calling any of the functions, make sure to stop generation after each function call and wait for it to be executed, before calling another function and continuing with your plan.

ENVIRONMENT SET-UP:
The pick point and the place point are given by two dimensional coordinates represented as [x-coordinate, y-coordinate].
The 2D coordinate system of the environment is as follows:
    1. The x-axis is in the horizontal direction, increasing to the right.
    2. The y-axis is in the vertical direction, increasing downwards.
    

"""

PROMPT_1 = \
"""
VARIABLES:
The following variables have already been defined. 
1. cloth_corners -> This is a numpy array that contains the corners of the cloth. It is an input to the identifyCornerRegions() function
2. cloth_center -> This is the center of the cloth.

CODE GENERATION:
Define a function called identifyCornerRegions() that identifies which corners lie at the top and which lie at the bottom, which ones are at towards the left and which ones towards the right. The function takes a numpy array as input and returns a list of tuples. Each tuple contains the corners and the region it belongs to.
Mark the code separately by ```python and ``` tags.
"""

PROMPT_2 = \
"""
VARIABLES:
The following variables have already been defined. 
1. cloth_corners -> This is a numpy array that contains the corners of the cloth. It is an input to the identifyCornerRegions() function
2. cloth_center -> This is the center of the cloth.

The user command is "[INSERT TASK]". Based on this information and the output of identifyCornerRegions(), define a function called identifyPickandPlace() which outputs the pick point and the place point. The cloth_corners and cloth_center are inputs to this function.
The pick point is stored in a variable called pick_point and the place point is stored in a variable place_point.
Mark the code separately by ```python and ``` tags.
"""

PROMPT_3 = \
"""
Based on the pick point and the place point, define a function findDistanceAndDirection() that returns a tuple containing the distance between the two points and angle between the two in radians. 
Mark the code separately by ```python and ``` tags.
"""

PROMPT_4 = \
"""
VARIABLES:
1. img_size -> This is an input to the function append_pixels_to_list
2. pick_point -> This is an input to the function append_pixels_to_list
3. place_point -> This is an input to the function append_pixels_to_list

CODE GENERATION:
Find the pick_point and place_point using the identifyPickandPlace() function.
Use append_pixels_to_list() function to append pick_point and place_point to the list. The first parameter is img_size, the second is pick_point and the third is place_point. 
Mark the code separately by ```python and ``` tags.
"""

PROMPT_5 = \
"""
VARIABLES:
1.depth -> This is an input to the get_world_coord_from_pixel function

CODE GENERATION:
Convert the 2-dimensional pick and place points into 3-dimensions using the get_world_coord_from_pixel function. The output is stored in pick_point_3d and place_point_3d
Mark the code separately by ```python and ``` tags.
"""

PROMPT_6 = \
"""
The output of get_world_coord_from_pixel function is given as input to the pick_and_place function. The third parameter of pick_and_place need not be specified.
Mark the code separately by ```python and ``` tags.
"""
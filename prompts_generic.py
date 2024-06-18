#####Mention which variables are already defined and what their use is


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
    
The robot arm needs to be moved to the first pick point. Ensure that the chosen pick point and place point lie inside the cloth. 

VARIABLES:
The following variables have already been defined. 
1. cloth_corners -> This is a numpy array that contains the corners of the cloth. It is an input to the identifyCornerRegions() function
2. cloth_center -> This is the center of the cloth.
3. camera_params -> This is an input to the function get_world_coord_from_pixel
4. img_size -> This is an input to the function append_pixels_to_list
5. test_pick_pixels -> This is an input to the function append_pixels_to_list
6. test_place_pixels -> This is an input to the function append_pixels_to_list
7. depth -> This is an input to the get_world_coord_from_pixel function

CODE GENERATION:
NOTE: Each fold consists of a series of sub-folds. Ensure that you generate code to achieve all the sub folds.
When generating the code to identify the pick and place points and fold the cloth as per requirements, do the following: 
1. Define a function called identifyCornerRegions() that identifies which corners lie at the top and which lie at the bottom, which ones are at towards the left and which ones towards the right. The function takes a numpy array as input and returns a list of tuples. Each tuple contains the corners and the region it belongs to.
2. The user command is "[INSERT TASK]". Based on this information and the output of identifyCornerRegions(), define a function called identifyPickandPlace() which outputs the pick point and the place point.  
3. Based on the pick point and the place point, define a function findDistanceAndDirection() that returns a tuple containing the distance between the two points and angle between the two in radians. 
4. Append the chosen pick and place point to the test_pick_pixel and test_place_pixel list using the append_pixels_to_list() function. Pass the first argument as args.img_size.
5. Convert the 2-dimensional pick and place points into 3-dimensions using the get_world_coord_from_pixel function
6. The output of get_world_coord_from_pixel function is given as input to the pick_and_place function. The third parameter of pick_and_place need not be specified.
7. For each of the above steps mark the code separately by ```python and ``` tags.

Ensure all eight steps are executed for each action

Finally, perform each of these steps one by one.

"""

ERROR_CORRECTION_PROMPT = \
"""Running code block [INSERT BLOCK NUMBER] of your previous response resulted in the following error:
[INSERT ERROR MESSAGE]
Can you output a modified code block to resolve this error? Ensure that you include all the functions that you generated previously along with the corrected version of the current code block.
"""

###deprecated

'''
10. When defining the functions, specify the required parameters, and document them clearly in the code. Make sure to include the orientation parameter.
11. If you want to print the calculated value of a variable to use later, make sure to use the print function to three decimal places, instead of simply writing the variable name. Do not print any of the trajectory variables, since the output will be too long.
INITIAL PLANNING 2:
Once the first fold is achieved, proceed to the second fold. Again, start by identifying the corners and the center of the cloth. Choose the pick and place points corresponding to the second fold.

INITIAL PLANNING 1:
Identify the regions [top left, top right, bottom left, bottom right] from which points need to be picked and placed in order to achieve the required fold. Think of the possible ways of folding a cloth and its effects.
Then, identify the pick point and place point based on this. 

INITIAL PLANNING 2:
Generate the pick and place points required to achieve the second fold. Ensure the fold is along the longest side. Stop generation after this step to wait until you obtain the printed outputs. 

Stop generation after each code block to wait for it to finish executing before continuing with your plan.
'''
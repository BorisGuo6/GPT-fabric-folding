import re
import numpy as np

system_prompt = '''**Cloth Folding Robot**
Role: You are the brain of a cloth folding robot. The robot would pick one spot on the cloth (referred to as the "pick point"), lift it by a small amount, drag it over to another spot (referred to as "the place point"), and finally release it.

Inputs:
- Method of folding: A description of how the cloth should be folded
- Cloth corners: The robot sees the cloth lying on a table from the top and gets a depth image for it. This depth image is then processed to extract the corner points for the cloth. The pixel co-ordinates for the corners will be given to you as an input. The format of each pixel coordinate would be [x-coordinate, y-coordinate]
- Initial cloth center: The robot will be given the [x-coordinate, y-coordinate] pair corresponding to the center of the initial cloth configuration

Task:
- Thought Process: Note down possible ways of picking and placing the cloth and their potential effects
- Planning: Provide a pair of pick and place point from the cloth corners provided as input for folding the cloth. 

Output:
- Planning (MOST IMPORTANT): Pick Point = (x 1, y 1) and Place Point = (x 2, y 2)
- Thought Process: Why did you choose these points and not something else?

PLEASE OUTPUT THE PICK POINT AND THE PLACE POINT FIRST AND THEN OUTPUT THE THOUGHT PROCESS INVOLVED
'''

def get_user_prompt(corners, center, task):
    '''
    This code consists of the specific CoT prompts designed for the different kinds of folds involved here    
    '''
    center = str(center)
    corners = str(corners)
    cloth_info = "- Cloth corners: " + corners + "\n- Initial Cloth center: " + center

    if task == "DoubleTriangle":
        return "- Method of folding: Choose two most distant points among the cloth corners and put them together to achieve a fold\n" + cloth_info
    elif task == "AllCornersInward":
        return "- Choose the point from the given cloth corners that is the FARTHEST from the Initial cloth center and return it as the Pick Point. Here, you MUST return the Initial cloth center as the Place point\n" + cloth_info
    else:
        return "Not implemented"
    
def parse_output(output):
    '''
    This function parses the string output returned by the LLM and returns the pick and place point coordinates that can be integrated in the code
    '''
    # Define regular expressions to match pick point and place point patterns
    pick_point_pattern = re.compile(r'Pick Point = \((\d+), (\d+)\)')
    place_point_pattern = re.compile(r'Place Point = \((\d+), (\d+)\)')

    # Use regular expressions to find matches in the text
    pick_match = pick_point_pattern.search(output)
    place_match = place_point_pattern.search(output)

    # Extract x and y values for pick point
    pick_point = None
    if pick_match:
        pick_point = np.array(tuple(map(int, pick_match.groups())))

    # Extract x and y values for place point
    place_point = None
    if place_match:
        place_point = np.array(tuple(map(int, place_match.groups())))

    return pick_point, place_point

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

image_analysis_instruction = '''I will be providing you with three images. In each image, you will see a background that's divided into four quadrants with alternating shades of gray. Think of this background as a flat surface on which a cloth is kept. 
This cloth could be seen in the centre of these images as an orange rectangle. The back of this cloth is pink in color and will not be visible in the first image where the cloth is lying flat on the table.

Now, starting from the initial configuration of the cloth as shown in the first image, someone picks a point on the cloth and places it to another point of the cloth achieving a fold. The result after this first folding step can be seen in the second image. And the cloth is folded again in a similar fashion till it reaches the configuration shown in the last image.

I want you to describe the instructions for each folding step (i.e from the first image to the second, from the second image to the third, and so on) that someone could follow to achieve the same fold.

NOTE: DO NOT INCLUDE WORDS LIKE ORANGE, TRIANGLE, SQUARE, PINK, LAYERS, EQUAL in the response.
Also, try to AVOID using directional references like top-right, bottom-left etc in the response. DESCRIBE THE ACTIONS IN TERMS OF THE CORNERS, CENTER, DIAGONAL DISTANCE etc.

RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:
- Fold 0: Instructions for the first fold
- Fold 1: Instructions for the second fold
...
- Fold n: Instructions for the last fold
'''

def get_user_prompt(corners, center, autoPrompt, instruction_list, index, task):
    '''
    This code consists of the specific CoT prompts designed for the different kinds of folds involved here    
    '''
    center = str(center)
    corners = str(corners)
    cloth_info = "- Cloth corners: " + corners + "\n- Initial Cloth center: " + center

    if autoPrompt:
        return instruction_list[index] + "\n" + cloth_info
    else:
        # The following lines correspond to the previous way of manually constructing the GPT prompts for the different folds
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

def analyze_images_gpt(path_list):
    '''
    This function takes the paths of the demonstration images and returns the description about what's happening
    TODO: Currently this function is only callable inside. Make sure that it could be imported while evaluating
    '''
    import base64
    from openai import OpenAI
    import requests
    api_key="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"
    client = OpenAI(api_key=api_key)

    # Function to encode image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    first_image = encode_image(path_list[0])
    second_image = encode_image(path_list[1])
    third_image = encode_image(path_list[2])
    # fourth_image = encode_image(path_list[3])
    # fifth_image = encode_image(path_list[4])

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_analysis_instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{first_image}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{second_image}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{third_image}"
                        }
                    }
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{fourth_image}"
                    #     }
                    # },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{fifth_image}"
                    #     }
                    # }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())

if __name__ == "__main__":
    # path_list = ["/home/ved2311/foldsformer/data/demo/AllCornersInward/rgb/0.png", "/home/ved2311/foldsformer/data/demo/AllCornersInward/rgb/1.png", "/home/ved2311/foldsformer/data/demo/AllCornersInward/rgb/2.png", "/home/ved2311/foldsformer/data/demo/AllCornersInward/rgb/3.png", "/home/ved2311/foldsformer/data/demo/AllCornersInward/rgb/4.png"]
    path_list = ["/home/ved2311/foldsformer/data/demo/DoubleTriangle/rgb/0.png", "/home/ved2311/foldsformer/data/demo/DoubleTriangle/rgb/1.png", "/home/ved2311/foldsformer/data/demo/DoubleTriangle/rgb/2.png"]
    analyze_images_gpt(path_list)
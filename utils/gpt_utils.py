import re
import numpy as np

system_prompt = '''**Cloth Folding Robot**
Role: You are the brain of a cloth folding robot. The robot would pick one spot on the cloth (referred to as the "pick point"), lift it by a small amount, drag it over to another spot (referred to as "the place point"), and finally release it.

Inputs:
- Method of folding: A description of how the cloth should be folded
- Cloth corners: The robot sees the cloth lying on a table from the top and gets a depth image for it. This depth image is then processed to extract the corner points for the cloth. The pixel co-ordinates for the corners will be given to you as an input. The format of each pixel coordinate would be [x-coordinate, y-coordinate]
- Cloth Center: The robot will be given the [x-coordinate, y-coordinate] pair corresponding to the center of the initial cloth configuration

Task:
- Thought Process: Note down possible ways of picking and placing the cloth and their potential effects
- Planning: Provide a pair of pick and place point from the cloth corners provided as input for folding the cloth. 

Output:
- Planning (MOST IMPORTANT): Pick Point = (x 1, y 1) and Place Point = (x 2, y 2)
- Thought Process: Why did you choose these points and not something else?

PLEASE OUTPUT THE PICK POINT AND THE PLACE POINT FIRST AND THEN OUTPUT THE THOUGHT PROCESS INVOLVED
'''

image_analysis_instruction = '''
I will be providing you with two images. In each image, you will see a background that's divided into four quadrants with alternating shades of gray. Think of this background as a flat surface on which a cloth is kept. 
This cloth could be seen in the centre of these images as a geometric shape coloured with orange and pink.

Can you see a black arrow in the first image provided to you? Can you point out where exactly is the tip of the black arrow located in all the images it is present in? 
DO NOT CONSIDER WHERE THE ARROW IS POINTING AT. ALSO DO NOT CONSIDER IN WHICH QUADRANT THE ARROW IS LOCATED AS THIS REASONING WILL CAUSE ERRORS. DO NOT USE THE WORD QUADRANT AT ALL. 
LOOK AT THE EXACT PIXEL LOCATION OF THE TIP OF THE ARROW. Is it near the CENTER? How close is it to the CENTER? It can RELATIVELY CLOSE TO THE CENTER OR QUITE FAR FROM THE CENTER. Explain your response briefly.
Now, consider the following task: 
If your answer to the image under consideration above indicated that the point is "somewhat close to the center" then return True or else return False.

This black arrow essentially represents an action where someone would pick a point on the cloth corresponding to the black dot from where the arrow originates, this would be represented as the picking point.
And point where the tip of the black arrow is located corresponds to the location where the chosen picking point is placed. This is referred to as the placing point.
This sequence of action of picking a point on the cloth and place it somewhere results in a fold, whose result can be seen in the next image. So basically we are folding the cloth in the first image to get to the second image.

I want you to describe the instructions for the folding step that someone could follow to achieve the same fold.
NOTE: DO NOT INCLUDE WORDS LIKE ORANGE, TRIANGLE, SQUARE, PINK, LAYERS, BLACK DOT in the response.
Also, try to AVOID using directional references like top-right, bottom-left etc in the response. DESCRIBE THE ACTIONS IN TERMS OF THE CORNERS, CENTER, DIAGONAL DISTANCE etc.

IMPORTANT: WHILE CHOOSING THE PLACING POINT FOR THE FOLDING ACTION, IF YOUR ANSWER TO THE PREVIOUS TRUE/FALSE QUESTION FOR THE CURRENT IMAGE IS TRUE THEN ALWAYS STATE THAT THE PLACING FOR THAT PARTICULAR FOLDING STEP WILL BE THE CENTER.

RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:
- Instructions: The instructions for the given folding step.

IMPORTANT: FOLLOW ALL THE GIVEN TASKS AND INSTRUCTIONS IN A SEQUENTIAL FASHION.
'''

def get_user_prompt(corners, center, autoPrompt, instruction, task):
    '''
    This code consists of the specific CoT prompts designed for the different kinds of folds involved here    
    '''
    center = str(center)
    corners = str(corners)
    cloth_info = "- Cloth corners: " + corners + "\n- Cloth center: " + center

    if autoPrompt:
        return instruction + "\n" + cloth_info
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

def analyze_images_gpt(image_list):
    '''
    This function takes the paths of the demonstration images and returns the description about what's happening
    TODO: Currently this function is only callable inside. Make sure that it could be imported while evaluating
    '''
    import base64
    import requests
    api_key="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"

    # Function to encode image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    first_image = encode_image(image_list[0])
    second_image = encode_image(image_list[1])

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    instruction = ""
    while instruction == "":
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
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        text = response['choices'][0]['message']['content']
        match = re.search(r'Instructions: ([^.]*\.)', text)
        if match:
            instruction = match.group(1)
    return instruction

if __name__ == "__main__":
    response = analyze_images_gpt(["data/demo/AllCornersInward/rgbviz/2.png", "data/demo/AllCornersInward/rgbviz/3.png"])
    print(response)
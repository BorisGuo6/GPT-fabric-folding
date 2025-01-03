import re
import numpy as np
import os 
import random
import json

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
I will be providing you with an image. In this image, you will see a background that's divided into four quadrants with alternating shades of gray. Think of this background as a flat surface on which a cloth is kept.
This cloth could be seen in the centre of these images with pink and yellow.
There is also a black arrow in the first image, which essentially represents an action where someone would pick a point on the cloth corresponding to the black dot from where the arrow originates. This would be represented as the picking point. On the other hand, the point where the tip of the black arrow is located corresponds to the location where the chosen picking point is placed. This is referred to as the placing point.

This sequence of action of picking a point on the cloth and place it somewhere results in a fold. So basically we are folding the cloth in the first image to get to the second image.
I want you to describe the instructions for the folding step that someone could follow to achieve the same fold. 
Look at the relative location of the tip of the arrow with respect to the center of the image. Depending on whether this is near the center or a diagonally opposite point , choose your placing point as the center or a diagonally opposite point respectively. 

the picking point is at the black dot and the placing point is at the tip of the arrow, find out the location relatived to cloth in detailed

IMPORTANT: INLCUDE THE PICKING AND PLACING POINT INFORMATION IN THE RESPONSE. YOU MUST SPECIFY WHERE SHOULD THE PLACING POINT BE.

IMPORTANT: Do not use terms like 'dot' or 'arrow' or 'horizontal' in the instructions. Instead, describe the spatial location using terms like 'bottom left' and 'top right'
Similarly, for identifying other corners, apply the same double-check of both coordinates. Ensure each chosen point meets both conditions (x and y) strictly to avoid incorrect selections.
                               
RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:
- Instructions: The instructions for the given folding step.
- Explanation: Why did you choose this pair of picking and placing points.
'''

# TODO - Remove the temp keys once the results for in-context learning are validated.
gpt_v_demonstrations = {
    "zero-shot":{
        "DoubleTriangle": {
            "data": [],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": []
        },
        "AllCornersInward": {
            "data": [],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": []
        },
        "CornersEdgesInward": {
            "data": [],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": []
        },
        "DoubleStraight": {
            "data": [],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": []
        },
        "Trousers": {
            "data": [],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": []
        }
    },
    "in-context":{
        "DoubleTriangle": {
            "data": ["7", "15", "19", "29", "33"],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": ["7", "15", "19", "33", "82", "29", "91", "66", "75", "59"],
            "gpt-demonstrations(temp)": ["5", "15", "35", "47", "82", "23", "91", "66", "75", "59"]
        },
        "AllCornersInward": {
            "data": [],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": ["4", "15", "35", "47", "82", "23", "91", "64", "75", "67"],
            "gpt-demonstrations(temp)": ["4", "15", "35", "47", "82", "23", "91", "64", "75", "67"]
        },
        "CornersEdgesInward": {
            "data": ["7", "15", "19", "29", "33"],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": ["7", "15", "19", "29", "33", "79", "84", "92", "27", "12"],
            "gpt-demonstrations(temp)": ["35", "40", "58", "67", "76", "79", "84", "92", "27", "12"]
        },
        "DoubleStraight": {
            "data": ["7", "15", "19", "29", "33"],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": ["7", "15", "33", "19", "82", "29", "91", "64", "75", "67"],
            "gpt-demonstrations(temp)": ["4", "15", "35", "47", "82", "23", "91", "64", "75", "67"]
        }
    }
}

def get_user_prompt(corners, center, autoPrompt, instruction, task, step):
    '''
    This code consists of the specific CoT prompts designed for the different kinds of folds involved here    
    '''
    center = str(center)
    corners = str(corners)
    cloth_info = "- Cloth corners: " + corners + "\n- Cloth center: " + center

    if autoPrompt:
        return instruction + "\n" + cloth_info
    else:
        # Getting the path to the root directory
        script_path = os.path.abspath(__file__)
        script_directory = os.path.dirname(script_path)

        # The following lines correspond to randomly selecting some of the possible instructions generated by GPT
        if task == "DoubleTriangle":
            parent_path = os.path.join(script_directory, "prompt-list", task, "Vision-ICL")
            step = max(0, step - 1)
            file_path = os.path.join(parent_path, str(step) + ".txt")
            with open(file_path, 'r') as f:
                instructions = f.readlines()
            return "- Method of folding: " + random.choice(instructions).strip() + "\n" + cloth_info
        elif task == "AllCornersInward":
            parent_path = os.path.join(script_directory, "prompt-list", task, "ZeroShot")
            step = max(0, step - 1)
            file_path = os.path.join(parent_path, str(step) + ".txt")
            with open(file_path, 'r') as f:
                instructions = f.readlines()
            return "- Method of folding: " + random.choice(instructions).strip() + "\n" + cloth_info
        elif task == "DoubleStraight":
            parent_path = os.path.join(script_directory, "prompt-list", task, "Vision-ICL")
            step = max(0, step - 1)
            file_path = os.path.join(parent_path, str(step) + ".txt")
            with open(file_path, 'r') as f:
                instructions = f.readlines()
            return "- Method of folding: " + random.choice(instructions).strip() + "\n" + cloth_info
        else:
            parent_path = os.path.join(script_directory, "prompt-list", task, "ZeroShot")
            step = max(0, step - 1)
            file_path = os.path.join(parent_path, str(step) + ".txt")
            with open(file_path, 'r') as f:
                instructions = f.readlines()
            return "- Method of folding: " + random.choice(instructions).strip() + "\n" + cloth_info
            
def parse_output(output):
    '''
    This function parses the string output returned by the LLM and returns the pick and place point coordinates that can be integrated in the code
    '''
    # Define regular expressions to match pick point and place point patterns
    pick_point_pattern = re.compile(r'Pick Point = \((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)')
    place_point_pattern = re.compile(r'Place Point = \((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)')

    # Use regular expressions to find matches in the text
    pick_match = pick_point_pattern.search(output)
    place_match = place_point_pattern.search(output)

    # Extract x and y values for pick point
    pick_point = None
    if pick_match:
        pick_point = np.array(np.round(tuple(map(float, pick_match.groups())))).astype(int)
    else:
        pick_point = np.array([None, None])

    # Extract x and y values for place point
    place_point = None
    if place_match:
        place_point = np.array(np.round(tuple(map(float, place_match.groups())))).astype(int)
    else:
        place_point = np.array([None, None])

    return pick_point, place_point

def analyze_images_gpt(image_list, task, action_id, eval_type, gpt_vision_model):
    '''
    This function takes the paths of the demonstration images and returns the description about what's happening
    '''
    import base64
    import requests
    # TODO: Use your own API key for performing the experiments
    api_key="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"

    # Function to encode image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    first_image = encode_image(image_list[0])
    second_image = encode_image(image_list[1])

    # Getting information corresponding to the demonstrations that we'd use
    gpt_vision_demonstrations_local = gpt_v_demonstrations[eval_type][task]

    # Getting the path to the in-context learning demonstrations for the model to analyze images better
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    parent_directory = os.path.dirname(script_directory)
    baseline_demo_path = os.path.join(parent_directory, "data", "gpt-4v-incontext-demonstrations", task)

    demo_images_list = gpt_vision_demonstrations_local["data"]
    demonstration_dictionary_list = []
    for demo_image_id in demo_images_list:
        demo_first_image = encode_image(os.path.join(baseline_demo_path, demo_image_id, "rgbviz", str(action_id) + ".png"))
        demo_second_image = encode_image(os.path.join(baseline_demo_path, demo_image_id, "rgbviz", str(action_id + 1) + ".png"))
        demo_instruction = gpt_vision_demonstrations_local["instruction"]
        user_prompt_dictionary = {
            "role": "user",
            "content": [
                {"type": "text", "text": image_analysis_instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{demo_first_image}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{demo_second_image}"
                    }
                }
            ]
        }
        assistant_response_dictionary = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": demo_instruction}
            ]
        }
        demonstration_dictionary_list += [user_prompt_dictionary, assistant_response_dictionary]

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    instruction = ""
    while instruction == "":
        payload = {
            "model": gpt_vision_model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be given some demonstration user prompts and the corresponding output that is expected from you.\
                                  Use these examples to guide your response for the actual query. \
                                Do not use terms like 'dot' or 'arrow' or 'horizontal' in the instructions. \
                                Instead, describe the spatial location using terms like 'bottom left' and 'bottom right.' \
                                "
                        }
                    ]
                }
            ] + demonstration_dictionary_list + 
            [
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
                        # {
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": f"data:image/jpeg;base64,{second_image}"
                        #     }
                        # }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # Check the status code and handle errors
        if response.status_code == 200:
            print("Request was successful!")
            response_data = response.json()
            print("Response data:", json.dumps(response_data, indent=4))
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")
            assert(0)
        response = response.json()
        if 'choices' in response:
            text = response['choices'][0]['message']['content']
            match = re.search(r'Instructions: ([^\n]+)\.', text)
            if match:
                instruction = match.group(1)
    return instruction + ".     For reference only:\
Choose a point that is located on the top left corner of the cloth relative to the center means to Choose a point that is strictly to the left and up the center (i.e., x-coordinate < center x, and y-coordinate < center y).\
Choose a point that is located on the bottom right corner of the cloth relative to the center means to Choose a point that is strictly to the right and below the center (i.e., x-coordinate > center x, and y-coordinate > center y).\
Choose a point that is located on the top right corner of the cloth relative to the center means to Choose a point that is strictly to the right and up the center (i.e., x-coordinate > center x, and y-coordinate < center y).\
Choose a point that is located on the bottom left corner of the cloth relative to the center means to Choose a point that is strictly to the left and below the center (i.e., x-coordinate < center x, and y-coordinate > center y).\
Similarly, for identifying other corners, apply the same double-check of both coordinates. Ensure each chosen point meets both conditions (x and y) strictly to avoid incorrect selections.\
Points that only partially meet these conditions should not be chosen."

if __name__ == "__main__":
    # Getting the responses for the first folding step
    response_set = set()
    for i in range(20):
        print(i)
        response = analyze_images_gpt(["data/demo/CornersEdgesInward/rgbviz/0.png", "data/demo/CornersEdgesInward/rgbviz/1.png"], "CornersEdgesInward", 0, "zero-shot", gpt_vision_model="gpt-4-vision-preview")
        response_set.add(response)
    response_list_1 = list(response_set)
    response_list_1 = sorted(response_list_1)
    response_list_1.append("\n")
    print()

    # Getting the responses for the second folding step
    response_set = set()
    for i in range(20):
        print(i)
        response = analyze_images_gpt(["data/demo/CornersEdgesInward/rgbviz/1.png", "data/demo/CornersEdgesInward/rgbviz/2.png"], "CornersEdgesInward", 1, "zero-shot", gpt_vision_model="gpt-4-vision-preview")
        response_set.add(response)
    response_list_2 = list(response_set)
    response_list_2 = sorted(response_list_2)
    response_list_2.append("\n")
    print()

    # Getting the responses for the third folding step
    response_set = set()
    for i in range(20):
        print(i)
        response = analyze_images_gpt(["data/demo/CornersEdgesInward/rgbviz/2.png", "data/demo/CornersEdgesInward/rgbviz/3.png"], "CornersEdgesInward", 2, "zero-shot", gpt_vision_model="gpt-4-vision-preview")
        response_set.add(response)
    response_list_3 = list(response_set)
    response_list_3 = sorted(response_list_3)
    response_list_3.append("\n")
    print()

    # Getting the responses for the fourth folding step
    response_set = set()
    for i in range(20):
        print(i)
        response = analyze_images_gpt(["data/demo/CornersEdgesInward/rgbviz/3.png", "data/demo/CornersEdgesInward/rgbviz/4.png"], "CornersEdgesInward", 3, "zero-shot", gpt_vision_model="gpt-4-vision-preview")
        response_set.add(response)
    response_list_4 = list(response_set)
    response_list_4 = sorted(response_list_4)
    response_list_4.append("\n")

    # Saving this data into a file
    response_list = response_list_1 + response_list_2 + response_list_3 + response_list_4
    with open("/home/ved2311/foldsformer/utils/prompt-list/corners_edges_inward.txt", "w") as f:
        for line in response_list:
            f.write(line + "\n")


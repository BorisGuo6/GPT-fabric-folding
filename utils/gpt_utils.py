import re
import numpy as np
import os 
import random
import cv2

###################
#Changed Version of the prompts
###################


system_prompt_bimanual = '''**Cloth Folding Robot**
Role: You are the brain of a cloth folding robot. The robot has two arms and it will use these two arms just as a human would. The robot picks the cloth at two places (referred to as the "pick points"), lifts them by a small amount, and drags them to two other spots (referred to as "the place points"), and finally releases the cloth from the two arms there.

Inputs:
- Method of folding: A description of how the cloth should be folded
- Cloth corners: The robot sees the cloth lying on a table from the top and gets a depth image for it. This depth image is then processed to extract the corner points for the cloth. The pixel co-ordinates for the corners will be given to you as an input. The format of each pixel coordinate would be [x-coordinate, y-coordinate]
- Cloth Center: The robot will be given the [x-coordinate, y-coordinate] pair corresponding to the center of the initial cloth configuration

Task:
- Thought Process: Note down possible ways of picking and placing the cloth and their potential effects. Remember that the robot can use both its arms to pick and place the cloth.
- Planning: Provide two pair of pick and place point from the cloth corners provided as input for folding the cloth. 

Output:
- Planning (MOST IMPORTANT): Pick Point 1 = (x 1, y 1) and Place Point 1 = (x 2, y 2) for the first arm and Pick Point 2 = (x 1', y 1') and Place Point 2 = (x 2', y 2')
- Thought Process: Why did you choose these points and not something else?

PLEASE OUTPUT THE TWO PICK POINT AND THE PLACE POINTs FIRST AND THEN OUTPUT THE THOUGHT PROCESS INVOLVED


'''
image_analysis_instruction_bimanual = ''' 
I will be providing you with two images. In each image, you will see a background that's divided into four quadrants with alternating shades of gray. Think of this background as a flat surface on which a cloth is kept.
This cloth could be seen in the centre of these images as a geometric shape coloured with blue and yellow.
The first image has its corners marked by black dots. These points are also numbered. 

This sequence of action of picking a point on the cloth and place it somewhere results in a fold, whose result can be seen in the next image. So basically we are folding the cloth in the first image to get to the second image.
I want you to describe the instructions for the folding step that someone could follow to achieve the same fold. 
IMPORTANT: INLCUDE ONE PICKING AND ONE PLACING POINT INFORMATION IN THE RESPONSE. YOU MUST SPECIFY WHERE SHOULD THE PLACING POINTS SHOULD BE.

RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:
- Instructions: The instructions for the given folding step and mention which point is located at the top.
- Explanation: Why did you choose this pairs of picking and placing points.
'''

image_analysis_instruction_new = ''' 
I will be providing you with two images. In each image, you will see a background that's divided into four quadrants with alternating shades of gray. Think of this background as a flat surface on which a cloth is kept.
This cloth could be seen in the centre of these images as a geometric shape coloured with blue and yellow.
The first image has its corners marked by black dots. These points are also numbered. From these points, identify the two points that are the top of the cloth and the two points that are at the bottom of the cloth. Be clear about the same. Think wisely. Also identify the two points that are on the lrft and the two that are on the right.

This sequence of action of picking a point on the cloth and place it somewhere results in a fold, whose result can be seen in the next image. Observe where the 4 regions (top left, bottom left, top right, bottom right) are located in the second image.
We are folding the cloth in the first image to get to the second image. Keep in mind which points of the first image correspond to which of the 4 regions. Observe which region is changing it's orientation or moving closer to other regions from the first image to the second. A point from that region is likely to be the pick point.
Check the second image to see which region is closest to the region with a changed orientation. This region is likely to contain the place point. 

I want you to describe the instructions for the folding step that someone could follow to achieve the same fold. 
If required you can use two pick and place points
IMPORTANT: INLCUDE ONE or TWO PICKING AND ONE or TWO PLACING POINT INFORMATION IN THE RESPONSE. YOU MUST SPECIFY WHERE SHOULD THE PLACING POINTS SHOULD BE.

RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:
- Instructions: The instructions for the given folding step and mention which point is located at the top.
- Explanation: Why did you choose these pairs of picking and placing points. Mention which points belong to which regions in the first image.
'''

#Remember that you have two hands. At any given instant you can use one hand or two hands simultaneously or one after the other in sequence. 

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

'''
"Mention which of the 4 corners is located at the top of cloth and which of the two corners is located at the bottom. 
Also mention which corners are on left and which ones are on right In a double triangle, one of the top corners has 
to align with the opposite bottom. That means the top right has to go to bottom left. Which of these points should 
be used to achieve this fold? RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:- Instructions: The instructions for the 
given folding step.- Explanation: Why did you choose this pair of picking and placing points. Which points are at the 
left which ones at the right. Also mention which ones are at top and which ones at bottom"
'''

image_analysis_instruction = '''
I will be providing you with two images. In each image, you will see a background that's divided into four quadrants with alternating shades of gray. Think of this background as a flat surface on which a cloth is kept.
This cloth could be seen in the centre of these images as a geometric shape coloured with orange and pink. The four corners of the image as well as the center is marked with black rectangles and labelled. The labels indicate which of the four regions the point belongs to: top left or bottom left or top right or bottom right. 
Observe how the position of the points belonging to each of these 4 regions changes from the first image to the second one.
There is also a black arrow in the first image, which essentially represents an action where someone would pick a point on the cloth corresponding to the black dot from where the arrow originates. This would be represented as the picking point. On the other hand, the point where the tip of the black arrow is located corresponds to the location where the chosen picking point is placed. This is referred to as the placing point.


This sequence of action of picking a point on the cloth and place it somewhere results in a fold, whose result can be seen in the next image. So basically we are folding the cloth in the first image to get to the second image.
I want you to describe the instructions for the folding step that someone could follow to achieve the same fold. 
Look at the relative location of the tip of the arrow with respect to the center of the image. Depending on whether this is near the center or a diagonally opposite point or a point along the given edge, choose your placing point as the center or a diagonally opposite point or a point along the given edge respectively. 

IMPORTANT: INLCUDE THE PICKING AND PLACING POINT INFORMATION IN THE RESPONSE. YOU MUST SPECIFY WHERE SHOULD THE PLACING POINT BE.

RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:
- Instructions: The instructions for the given folding step.
- Explanation: Why did you choose this pair of picking and placing points. 
'''

gpt_v_demonstrations = {
    "zero-shot":{
        "DoubleTriangle": {
            "data": [],
            "instruction": "- Use this pair of images to guide your response. Which are the two points at the top. Mention this separately.",
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
        }
    },
    "in-context":{
        "DoubleTriangle": {
            "data": ["7", "15", "19", "29", "33"],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": ["5", "15", "35", "47", "82", "23", "91", "66", "75", "59"]
        },
        "AllCornersInward": {
            "data": [],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": ["4", "15", "35", "47", "82", "23", "91", "64", "75", "67"]
        },
        "CornersEdgesInward": {
            "data": ["7", "15", "19", "29", "33"],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": ["35", "40", "58", "67", "76", "79", "84", "92", "27", "12"]
        },
        "DoubleStraight": {
            "data": ["7", "15", "19", "29", "33"],
            "instruction": "- Use this pair of images to guide your response",
            "gpt-demonstrations": ["4", "15", "35", "47", "82", "23", "91", "64", "75", "67"]
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
        # The following lines correspond to randomly selecting some of the possible instructions generated by GPT
        if task == "DoubleTriangle":
            parent_path = "/home/ved2311/foldsformer/utils/prompt-list/DoubleTriangle/Vision-ICL"
            step = max(0, step - 1)
            file_path = os.path.join(parent_path, str(step) + ".txt")
            with open(file_path, 'r') as f:
                instructions = f.readlines()
            return "- Method of folding: " + random.choice(instructions).strip() + "\n" + cloth_info
        elif task == "AllCornersInward":
            parent_path = "/home/ved2311/foldsformer/utils/prompt-list/AllCornersInward/ZeroShot"
            step = max(0, step - 1)
            file_path = os.path.join(parent_path, str(step) + ".txt")
            with open(file_path, 'r') as f:
                instructions = f.readlines()
            return "- Method of folding: " + random.choice(instructions).strip() + "\n" + cloth_info
        elif task == "DoubleStraight":
            parent_path = "/home/ved2311/foldsformer/utils/prompt-list/DoubleStraight/Vision-ICL"
            step = max(0, step - 1)
            file_path = os.path.join(parent_path, str(step) + ".txt")
            with open(file_path, 'r') as f:
                instructions = f.readlines()
            return "- Method of folding: " + random.choice(instructions).strip() + "\n" + cloth_info
        else:
            parent_path = "/home/ved2311/foldsformer/utils/prompt-list/CornersEdgesInward/ZeroShot"
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


def set_of_marks(image_path, corners, center = None, regions = False, path = False):
    if path:
        image = cv2.imread(image_path)
    else:
        image =image_path
    if center!=None:
        corners = np.append(corners, [list(center)], axis = 0)
    print(corners)
    for i, pt in enumerate(corners, start = 1):
        cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0, 0, 0), -1)
        cv2.putText(image, str(i), (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        if regions:
            if i == 1:
                cv2.putText(image, "bottom", (int(pt[0])-35, int(pt[1])-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, "right", (int(pt[0])-35, int(pt[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            elif i == 2:
                cv2.putText(image, "bottom left", (int(pt[0])-15, int(pt[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            elif i == 3:
                cv2.putText(image, "top right", (int(pt[0])-35, int(pt[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, "top left", (int(pt[0])-15, int(pt[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
    
    return image

def parse_output_bimanual(output):
    '''
    This function parses the string output returned by the LLM and returns the pick and place point coordinates that can be integrated in the code
    '''
    # Define regular expressions to match pick point and place point patterns
    pick_point_pattern_1 = re.compile(r'Pick Point 1 = \((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)')
    place_point_pattern_1 = re.compile(r'Place Point 1 = \((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)')
    
    pick_point_pattern_2 = re.compile(r'Pick Point 2 = \((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)')
    place_point_pattern_2 = re.compile(r'Place Point 2 = \((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)')

    # Use regular expressions to find matches in the text
    pick_match_1 = pick_point_pattern_1.search(output)
    place_match_1 = place_point_pattern_1.search(output)
    
    pick_match_2 = pick_point_pattern_2.search(output)
    place_match_2 = place_point_pattern_2.search(output)

    # Extract x and y values for pick point
    pick_point_1 = None
    pick_point_2 = None
    if pick_match_1 and pick_match_2:
        pick_point_1 = np.array(np.round(tuple(map(float, pick_match_1.groups())))).astype(int)
        pick_point_2 = np.array(np.round(tuple(map(float, pick_match_2.groups())))).astype(int)
    else:
        pick_point_1 = np.array([None, None])
        pick_point_2 = np.array([None, None])

    # Extract x and y values for place point
    place_point = None
    if place_match_1 and place_match_2:
        place_point_1 = np.array(np.round(tuple(map(float, place_match_1.groups())))).astype(int)
        place_point_2 = np.array(np.round(tuple(map(float, place_match_2.groups())))).astype(int)
    else:
        place_point_1 = np.array([None, None])
        place_point_2 = np.array([None, None])

    return pick_point_1, place_point_1, pick_point_2, place_point_2


def analyze_images_gpt(image_list, task, action_id, eval_type):
    '''
    This function takes the paths of the demonstration images and returns the description about what's happening
    '''
    import base64
    import requests
    # TODO - Remove the API key before making the code public
    api_key="sk-proj-gpjnKOl4bOwfXhGQQfbVT3BlbkFJUe0DAyVZTe1G6oKNubnD"

    # Function to encode image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    first_image = encode_image(image_list[0])
    second_image = encode_image(image_list[1])

    # Getting information corresponding to the demonstrations that we'd use
    gpt_vision_demonstrations_local = gpt_v_demonstrations[eval_type][task]
    baseline_demo_path = os.path.join("/home/ved2311/foldsformer-baseline/data/demonstrations", task)
    demo_images_list = gpt_vision_demonstrations_local["data"]
    demonstration_dictionary_list = []
    
    #the for loop won't be executed in the context of zeroshot learning
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
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be given some demonstration user prompts and the corresponding output that is expected from you. Use these examples to guide your response for the actual query."
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
            "temperature": 0
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        print(response)
        if 'choices' in response:
            text = response['choices'][0]['message']['content']
            match = re.search(r'Instructions: ([^\n]+)\.', text)
            if match:
                instruction = match.group(1)
    return instruction

if __name__ == "__main__":
    # Getting the responses for the first folding step
    response_set = set()
    for i in range(20):
        print(i)
        response = analyze_images_gpt(["data/demo/CornersEdgesInward/rgbviz/0.png", "data/demo/CornersEdgesInward/rgbviz/1.png"], "CornersEdgesInward", 0, "zero-shot")
        response_set.add(response)
    response_list_1 = list(response_set)
    response_list_1 = sorted(response_list_1)
    response_list_1.append("\n")
    print()

    # Getting the responses for the second folding step
    response_set = set()
    for i in range(20):
        print(i)
        response = analyze_images_gpt(["data/demo/CornersEdgesInward/rgbviz/1.png", "data/demo/CornersEdgesInward/rgbviz/2.png"], "CornersEdgesInward", 1, "zero-shot")
        response_set.add(response)
    response_list_2 = list(response_set)
    response_list_2 = sorted(response_list_2)
    response_list_2.append("\n")
    print()

    # Getting the responses for the third folding step
    response_set = set()
    for i in range(20):
        print(i)
        response = analyze_images_gpt(["data/demo/CornersEdgesInward/rgbviz/2.png", "data/demo/CornersEdgesInward/rgbviz/3.png"], "CornersEdgesInward", 2, "zero-shot")
        response_set.add(response)
    response_list_3 = list(response_set)
    response_list_3 = sorted(response_list_3)
    response_list_3.append("\n")
    print()

    # Getting the responses for the fourth folding step
    response_set = set()
    for i in range(20):
        print(i)
        response = analyze_images_gpt(["data/demo/CornersEdgesInward/rgbviz/3.png", "data/demo/CornersEdgesInward/rgbviz/4.png"], "CornersEdgesInward", 3, "zero-shot")
        response_set.add(response)
    response_list_4 = list(response_set)
    response_list_4 = sorted(response_list_4)
    response_list_4.append("\n")

    # Saving this data into a file
    response_list = response_list_1 + response_list_2 + response_list_3 + response_list_4
    with open("/home/ved2311/foldsformer/utils/prompt-list/corners_edges_inward.txt", "w") as f:
        for line in response_list:
            f.write(line + "\n")


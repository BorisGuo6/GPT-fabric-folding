import re
import numpy as np
import os 
import random
import cv2

image_analysis_instruction_bimanual = ''' 
I will be providing you with two images. In each image, you will see a background that's divided into four quadrants with alternating shades of gray. Think of this background as a flat surface on which a cloth is kept.
This cloth could be seen in the centre of these images as a geometric shape coloured with blue and yellow.
The first image has its corners and center marked by black dots. These points are also numbered. 

This sequence of action of picking a point on the cloth and place it somewhere results in a fold, whose result can be seen in the next image. So basically we are folding the cloth in the first image to get to the second image.
I want you to describe the instructions for the folding step that someone could follow to achieve the same fold. 
IMPORTANT: INLCUDE ONE PICKING AND ONE PLACING POINT INFORMATION IN THE RESPONSE. YOU MUST SPECIFY WHERE SHOULD THE TWO PLACING POINTS BE.

RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:
- Instructions: The instructions for the given folding step.
- Explanation: Why did you choose these two pairs of picking and placing points.
'''

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

def set_of_marks(image, corners, center = None):
    if center!=None:
        corners = np.append(corners, [list(center)], axis = 0)
    print(corners)
    for i, pt in enumerate(corners, start = 1):
        cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0, 0, 0), -1)
        cv2.putText(image, str(i), (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return image

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
                {"type": "text", "text": image_analysis_instruction_bimanual},
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
                        {"type": "text", "text": image_analysis_instruction_bimanual},
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
        if 'choices' in response:
            text = response['choices'][0]['message']['content']
            match = re.search(r'Instructions: ([^\n]+)\.', text)
            if match:
                instruction = match.group(1)
    return instruction

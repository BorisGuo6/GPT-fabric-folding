import os
import os.path as osp
import argparse
import time
import json
import base64
import re
import requests
from openai import OpenAI
import csv
import cv2 as cv
import math
import numpy as np

import datetime
from collections import deque
from matplotlib import pyplot as plt
from PIL import Image
import supervision as sv

import torch

CHECKPOINT_PATH='sam_vit_h_4b8939.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE="vit_h"

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)


mask_generator = SamAutomaticMaskGenerator(sam)

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv.imwrite("/home/rajeshgayathri2003/GPT-fabric-folding/goals/square_towel/mask_image.jpg", mask_image)
    #ax.imshow(mask_image)
    
#def show_points(coords, labels, marker_size=375):
    #pos_points = coords[labels==1]
    #neg_points = coords[labels==0]
    #ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    #ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def preprocess(img):
    img_copy=img.copy()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    blur = cv.medianBlur(gray,5)
    th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    

    
    # Find contours from the binary image
    contours, _ = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    possible_contours=[]
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # Draw the bounding box on the original image
        if w<400 and w>50 and h<400 and h>50:
            possible_contours.append(contour)
            
            
            
    if len(possible_contours)!=1:
        best_contour=max(possible_contours, key=cv.contourArea)
    else:
        best_contour=possible_contours[0]
        
    x, y, w, h = cv.boundingRect(best_contour)    
    
    print(f"opencv bbox {x},{y},{w},{h}.")

        
    
    
    new_x=int(x+0.5*w)
    new_y=int(y+0.5*h)
    cv.circle(img_copy,(new_x,new_y),4,(255,255,255),-1)
    center_point=[[new_x,new_y]]
    
    cv.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 255), 1)
    
    coverage_pix=cv.contourArea(best_contour)
    corner = [[x,y]]
    
    return img_copy,coverage_pix,center_point, corner



def preprocess_2(img,pixel):
    img_copy=img.copy()
    img_rgb=cv.cvtColor(img_copy,cv.COLOR_BGR2RGB)
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(img_rgb)
    '''result = mask_generator.generate(img)
    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(img, detections)
    '''
    print(pixel)
    
    input_point=np.array(pixel)
    input_label=np.array([1])
    
    
    masks, scores, logits = mask_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
    )

    area=np.sum(masks[0])
    print(scores[0])
    print(np.sum(masks[0]))
      
    return img_rgb,area


folder="/home/rajeshgayathri2003/GPT-fabric-folding/goals/square_towel"


s_max_image="start.png"

s_0_image="test_goal_2.png"

s_f_image="test_goal_3.png"


max_img=cv.imread(osp.join(folder,s_max_image))
ini_img=cv.imread(osp.join(folder,s_0_image))
final_img=cv.imread(osp.join(folder,s_f_image))

img_goal,max_coverage,center_point, corner=preprocess(max_img)
img1,max_coverage_2=preprocess_2(max_img,center_point)
cv.imwrite("/home/rajeshgayathri2003/GPT-fabric-folding/goals/square_towel/mask_image_1.jpg", img1)  
cv.imwrite(osp.join(folder,"goal_processed.png"),img_goal)

img_ini,ini_coverage,center_point, corner=preprocess(ini_img)
img2,ini_coverage_2=preprocess_2(ini_img,corner)
cv.imwrite("/home/rajeshgayathri2003/GPT-fabric-folding/goals/square_towel/mask_image_2.jpg", img2)

cv.imwrite(osp.join(folder,"ini_processed.png"),img_ini)

img_final,final_coverage,center_point,corner=preprocess(final_img)
img3,final_coverage_2=preprocess_2(final_img,corner)
cv.imwrite("/home/rajeshgayathri2003/GPT-fabric-folding/goals/square_towel/mask_image_3.jpg", img3)
cv.imwrite(osp.join(folder,"final_processed.png"),img_final)


print(f"The max coverage is {max_coverage}, the initial coverage is {ini_coverage}, the final coverage is {final_coverage}. So the Normalized Improvement is {(final_coverage-ini_coverage)/(max_coverage-ini_coverage)}")
print(f"The max coverage is {max_coverage_2}, the initial coverage is {ini_coverage_2}, the final coverage is {final_coverage_2}. So the Normalized Improvement is {(final_coverage_2-ini_coverage_2)/(max_coverage_2-ini_coverage_2)}")

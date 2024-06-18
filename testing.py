from slurm_utils import find_corners, find_pixel_center_of_cloth, append_pixels_to_list
from datetime import date, timedelta
import os
import numpy as np
date_today = date.today()

image_path = os.path.join("eval result", "DoubleTriangle", "square", str(date_today), str(0), str(0), "depth", str(0) + ".png")

code3= """
```python
global pick_point_3d
# Find the center of the cloth
center = find_pixel_center_of_cloth(image_path, False)
print(f"Center of the cloth: {center}")

# Find the corners of the cloth
corners = find_corners(image_path, False)
print(f"Corners of the cloth: {corners}")

# Identify the top and bottom corners
top_corners = [corner for corner in corners if corner[1] < center[1]]
bottom_corners = [corner for corner in corners if corner[1] > center[1]]

# Choose one top corner as the pick point and one bottom corner as the place point
pick_point = top_corners[0]
place_point = bottom_corners[0]

print(f"Pick point: {pick_point}")
print(f"Place point: {place_point}")
```

## Initial Planning 2
Now that we have identified the pick and place points, we can generate the pick and place points required to achieve the first fold. We will then convert these 2D points into 3D points and perform the pick and place action. After the action is completed, we will render the environment and save the images.

```python
# Append the pick and place points to the lists
append_pixels_to_list(128, pick_point, place_point, [], [])

```

"""
code = """
import math
def find_distance(x,y):
    return x-y

def find_real_distance(p1, p2):
    val1 = find_distance(p1[0], p2[0])
    val2 = find_distance(p1[1], p2[1])
    return math.sqrt(val1**2+val2**2)

print(find_real_distance((5,6), (7,9)))

"""
code2 = """
import numpy as np
# Find the corners of the cloth
#corners = find_corners(image_path, False)
corners = np.array([[100,100], [100,27], [27,100], [27,27]])


# Define a function to find the two corners that are farthest apart
def find_farthest_corners(corners):
    max_distance = 0
    farthest_corners = None
    for i in range(len(corners)):
        for j in range(i+1, len(corners)):
            distance = find_real_distance(corners[i], corners[j])
            if distance > max_distance:
                max_distance = distance
                farthest_corners = (corners[i], corners[j])
    return farthest_corners

# Find the two corners that are farthest apart
farthest_corners = find_farthest_corners(corners)
print(farthest_corners)
"""

cloth_corners = np.array([[100,100], [27,27], [100,27], [27,100]])
img_size =128
cloth_center = (63,63)

code = open('log_0.txt', 'r')
code = code.read()
code_block = code.split("```python")

# for block in code_block:
#     if len(block.split("```")) > 1:
#         code = block.split("```")[0]
    
#         if code:
#             exec(code)
#             print("done")

def test():
    lcls = locals()
    exec( 'a = 3', lcls)
    
    print(f'a is {a}')
test()
'''
This script has been adapted from Kai Mo's scripts for Foldsformer, along with some additional changes to extend its functionality 
'''
import numpy as np
import random

class DoubleTriangle:
    def __init__(self):
        self.pick_idxs = [[1,0],[1,3],[2,0],[2,3],[0,1],[0,2],[3,1],[3,2]]

    def get_action(self, curr_corners, pick_idx):
        if pick_idx == 2 or pick_idx == 3:
            place_idx = 3 - pick_idx
        elif pick_idx == 1:
            place_idx = 2
        else:
            place_idx = 3
        pick_pos = curr_corners[pick_idx]
        place_pos = curr_corners[place_idx]

        return pick_pos, place_pos


class AllCornersInward:
    def __init__(self):
        self.gamma = [0.9, 0.95, 1.0]

    def get_action(self, curr_corners, center, pick_idx, center_place):
        pick_pos = curr_corners[pick_idx]
        
        options = [0, 1, 2]
        gamma_index = 2 if center_place else random.choice(options)
        place_pos = pick_pos + self.gamma[gamma_index] * (center - pick_pos)

        return pick_pos, place_pos


class DoubleStraight:
    def __init__(self):
        zero_list = [[(0, 1), (2, 3), (4, 7)], [(0, 1), (2, 3), (7, 4)], [(0, 2), (1, 3), (5, 6)], [(0, 2), (1, 3), (6, 5)]]
        one_list = [[(1, 0), (3, 2), (4, 7)], [(1, 0), (3, 2), (7, 4)], [(1, 3), (0, 2), (5, 6)], [(1, 3), (0, 2), (6, 5)]]
        two_list = [[(2, 0), (3, 1), (5, 6)], [(2, 0), (3, 1), (6, 5)], [(2, 3), (0, 1), (4, 7)], [(2, 3), (0, 1), (7, 4)]]
        three_list = [[(3, 1), (2, 0), (5, 6)], [(3, 1), (2, 0), (6, 5)], [(3, 2), (1, 0), (4, 7)], [(3, 2), (1, 0), (7, 4)]]
        self.pickplace_idxs = zero_list + one_list + two_list + three_list

    def get_action(self, curr_corners, edge_middles, pickplace_idx):
        all_keypoints = np.concatenate((curr_corners, edge_middles), axis=0)
        pick_pos, place_pos = all_keypoints[pickplace_idx[0]], all_keypoints[pickplace_idx[1]]
        return pick_pos, place_pos


class CornersEdgesInward:
    def __init__(self):
        zero_list = [[(0, 8), (1, 8), (2, 5), (3, 6)], [(0, 8), (1, 8), (3, 6), (2, 5)], [(0, 8), (2, 8), (1, 4), (3, 7)], [(0, 8), (2, 8), (3, 7), (1, 4)]]
        one_list = [[(1, 8), (0, 8), (2, 5), (3, 6)], [(1, 8), (0, 8), (3, 6), (2, 5)], [(1, 8), (3, 8), (0, 4), (2, 7)], [(1, 8), (3, 8), (2, 7), (0, 4)]]
        three_list = [[(3, 8), (1, 8), (0, 4), (2, 7)], [(3, 8), (1, 8), (2, 7), (0, 4)], [(3, 8), (2, 8), (0, 5), (1, 6)], [(3, 8), (2, 8), (1, 6), (0, 5)]]
        two_list = [[(2, 8), (0, 8), (1, 4), (3, 7)], [(2, 8), (0, 8), (3, 7), (1, 4)], [(2, 8), (3, 8), (0, 5), (1, 6)], [(2, 8), (3, 8), (1, 6), (0, 5)]]
        self.pickplace_idxs = zero_list + one_list + three_list + two_list
        self.gamma = 1.0

    def get_action(self, curr_corners, edge_middles, center, pickplace_idx):
        all_keypoints = np.concatenate((curr_corners, edge_middles, np.array([center])), axis=0)
        pick_pos = all_keypoints[pickplace_idx[0]]
        target_pos = all_keypoints[pickplace_idx[1]]
        place_pos = pick_pos + self.gamma * (target_pos - pick_pos)

        return pick_pos, place_pos


Demonstrator = {
    "DoubleTriangle": DoubleTriangle,
    "DoubleStraight": DoubleStraight,
    "AllCornersInward": AllCornersInward,
    "CornersEdgesInward": CornersEdgesInward,
}

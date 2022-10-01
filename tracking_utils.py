import pickle
from typing import Any, Mapping
import numpy as np

from file_sys_utils import file_open, file_read, get_path_to_file
from general_utils import ListType


def get_ground_truth_tracking(path_dict: str, idx: int):

    '''
    Return path, pointer, ground truth coordinates for the x and y coordinates, offsets, and image background of the dataset at path_dict[idx].

    Args:
        path_dict: dictionary for dataset access with custom file_open function and h5py
        idx: index for indexing the dataset

    Returns:
        path_to_file: path to the data file
        handle: integer that is used as pointer to the file for data access
        ground_truth_x: list of ground truth x coordinates in absolute coordinates
        ground_truth_y: list of ground truth y coordinates in absolute coordinates
        offset_x: list of x coordinates offsets
        offset_y: list of y coordinates offsets
        x_background: background image 
    '''

    path_to_file = get_path_to_file(path_dict, idx)
    handle = file_open(path_to_file)
    train_data = pickle.load(open(path_to_file, 'rb'))
    ground_truth_x, ground_truth_y, offset_x, offset_y = train_data['ground_truth']
    ground_truth_x_fov = ground_truth_x - offset_x
    ground_truth_y_fov = ground_truth_y - offset_y
    x_background = train_data['background']

    return path_to_file, handle, ground_truth_x, ground_truth_y, offset_x, offset_y, x_background

def update_anchor_coordinates(args: Mapping[str, Any],
                              x_anchor: int,
                              y_anchor: int,
                              coordinate_x: int,
                              coordinate_y: int):

    '''
    Updates the anchor/offset coordinates, given the estimated specimen coordinate

    Args:
        args: non-tunable hyperparameters
        x_anchor: x coordinate of left upper corner of the fov in absolute coordinates, plays the role of offset_x, which is the ground truth anchor coordinate
        y_anchor: y coordinate of left upper corner of the fov in absolute coordinates, plays the role of offset_y, which is the ground truth anchor coordinate
        coordinate_x: estimated x coordinate
        coordinate_y: estimated y coordinate

    Returns:
        x_anchor: updated x coordinate of left upper corner of the fov in absolute coordinates
        y_anchor: updated y coordinate of left upper corner of the fov in absolute coordinates
    '''

    x_anchor_diff = coordinate_x - int(args.h/2)
    y_anchor_diff = coordinate_y - int(args.h/2)
    x_anchor = x_anchor + x_anchor_diff
    y_anchor = y_anchor + y_anchor_diff
    return x_anchor, y_anchor

def get_x_full(handle: int,
               x_background: ListType, 
               offset_x: int,
               offset_y: int,
               i: int):

    '''
    Return background with specimen at frame index i 

    Args:
        handle: pointer to data file
        x_background: background image 
        offset_x: list of x coordinates offsets
        offset_y: list of y coordinates offsets
        i: frame index

    Returns:
        x_full: background with specimen
    '''

    x_full = np.copy(x_background)
    x = file_read(handle, i)
    x_full[offset_y[i]:offset_y[i]+512, offset_x[i]:offset_x[i]+512] = x
    return x_full

def get_x_tracking(args: Mapping[str, Any],
                   handle: int,
                   x_background: ListType,
                   offset_x: int,
                   offset_y: int,
                   x_anchor: int,
                   y_anchor: int,
                   i: int,
                   peak: int = 4095):

    '''
    Similar to get_x, using offset values from simulated live tracking

    Args:
        args: non-tunable hyperparameters
        handle: pointer to data file
        x_background: background image 
        offset_x: list of x coordinates offsets
        offset_y: list of y coordinates offsets
        x_anchor: x coordinate of left upper corner of the fov in absolute coordinates
        y_anchor: y coordinate of left upper corner of the fov in absolute coordinates
        i: frame index
        peak: normalization constant

    Returns:
        x: 512 * 512 fov with estimated coordinate at the center. Same as 
    '''

    x_full = get_x_full(handle, x_background, offset_x, offset_y, i)
    x = np.zeros((args.h, args.h))
    x_fov = x_full[np.clip(y_anchor, 0, args.H):np.clip(y_anchor+args.h, 0, args.H), np.clip(x_anchor, 0, args.W):np.clip(x_anchor+args.h, 0, args.W)]
    x[np.clip(-y_anchor, 0, args.H):np.clip(args.H - y_anchor, 0, args.h), np.clip(-x_anchor, 0, args.W):np.clip(args.W - x_anchor, 0, args.h)] = x_fov
    x = x/peak
    return x
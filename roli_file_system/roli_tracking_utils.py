import h5py, pickle
from typing import Any, Mapping
import numpy as np

from file_sys_utils import file_open, file_read, get_path_to_file
from general_utils import ListType


def get_ground_truth_tracking(path_dict: str, idx: int):

    '''
    Return ground truth coordinates for the x and y coordinates in the field of view (fov) over the entire dataset,
    and exclude indices where tracking quality is low.

    Args:
        path_dict: dictionary for dataset access with custom file_open function and h5py
        idx: index for indexing the dataset

    Returns:
        path_to_file: path to the data file
        handle: integer that is used as pointer to the file for data access
        ground_truth_x_fov: list of ground truth x coordinates in absolute coordinates
        ground_truth_y_fov: list of ground truth y coordinates in absolute coordinates
        offset_x: list of x coordinates offsets
        offset_y: list of y coordinates offsets
        x_background: background image 
    '''

    path_to_file = get_path_to_file(path_dict, idx)
    handle = file_open(path_to_file.encode('utf-8'))
    ds = h5py.File(get_path_to_file(path_dict, idx, 'behavior.h5'), 'r')
    
    if 'fish_yolk_x' in ds.keys():
        ground_truth_x = np.array(ds['fish_yolk_x'])
        ground_truth_y = np.array(ds['fish_yolk_y'])
        offset_x = np.array(ds['offset_x'])
        offset_y = np.array(ds['offset_y'])
        
    elif 'x_fish' in ds.keys():
        ground_truth_x = np.array(ds['x_fish'])
        ground_truth_y = np.array(ds['y_fish'])
        offset_x = np.array(ds['x_offset'])
        offset_y = np.array(ds['y_offset'])

    x_background = np.array(ds['img_bg'])[-1]

    return path_to_file, handle, ground_truth_x, ground_truth_y, offset_x, offset_y, x_background


def update_anchor_coordinates(args: Mapping[str, Any],
                              x_anchor: int,
                              y_anchor: int,
                              prior_x: int,
                              prior_y: int):
    x_anchor_diff = prior_x - int(args.h/2)
    y_anchor_diff = prior_y - int(args.h/2)
    x_anchor = x_anchor + x_anchor_diff
    y_anchor = y_anchor + y_anchor_diff
    return x_anchor, y_anchor


def get_x_full(handle: int,
               x_background: ListType, 
               offset_x: int,
               offset_y: int,
               i: int):
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
    x_full = get_x_full(handle, x_background, offset_x, offset_y, i)
    x = np.zeros((args.h, args.h))
    x_fov = x_full[np.clip(y_anchor, 0, args.H):np.clip(y_anchor+args.h, 0, args.H), np.clip(x_anchor, 0, args.W):np.clip(x_anchor+args.h, 0, args.W)]
    x[np.clip(-y_anchor, 0, args.H):np.clip(args.H - y_anchor, 0, args.h), np.clip(-x_anchor, 0, args.W):np.clip(args.W - x_anchor, 0, args.h)] = x_fov
    x = x/peak
    return x
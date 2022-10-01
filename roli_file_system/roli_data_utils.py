import pickle
from typing import Any, List, Mapping, Tuple
import h5py

import numpy as np
import torch as pt
import torchvision

from file_sys_utils import *

__all__ = ['get_x',
           'get_target',
           'get_ground_truth',
           'get_ground_truth_cross_val',
           'get_ground_truth_tracking']

gaussian_blur = torchvision.transforms.GaussianBlur(9, 4.0)

def get_x(handle: int, i: int, peak: int = 4095) -> np.ndarray:
    '''
    Return normalized input frame

    Args:
        handle: integer that is used as pointer to the file for data access
        i: frame index

    Returns: 
        x: normalized frame
    '''
    x = file_read(handle, i)
    x = x/peak
    return x


def get_target(args: Mapping[str, Any], x_coordinate: int, y_coordinate: int) -> np.ndarray:
    target = np.zeros((args.h, args.h))
    target[y_coordinate, x_coordinate] = 1
    target = pt.Tensor(target.astype(np.float32)).reshape(1, 1, args.h, args.h)
    target = gaussian_blur(target)
    target = target/pt.max(target)
    target = target[0, 0].cpu().detach().numpy()
    return target


def get_ground_truth(path_dict: Mapping[str, Any], idx: int) -> Tuple[List[int], ...]:

    '''
    Return ground truth coordinates for the x and y coordinates in the field of view (fov) over the entire dataset,
    and exclude indices where tracking quality is low.

    Args:
        path_dict: dictionary for dataset access with custom file_open function and h5py
        idx: index for indexing the dataset

    Returns:
        path_to_file: path to the data file
        handle: integer that is used as pointer to the file for data access
        indices: list of indices indexing the dataset and the ground truth coordinates
        ground_truth_x_fov: list of ground truth x coordinates
        ground_truth_y_fov: list of ground truth y coordinates
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
        
    ground_truth_x_fov = ground_truth_x - offset_x
    ground_truth_y_fov = ground_truth_y - offset_y

    correlation_coefficient = np.array(ds['C'])
    indices = np.array(range(len(correlation_coefficient)))
    exclude_indices = np.concatenate([np.where(ground_truth_x_fov < 0)[0],
                                      np.where(ground_truth_y_fov < 0)[0],
                                      np.where(correlation_coefficient < .75)[0],
                                      np.where(correlation_coefficient > 1.0)[0]])
    indices = np.delete(indices, np.unique(exclude_indices))

    return path_to_file, handle, indices, ground_truth_x_fov, ground_truth_y_fov


def get_ground_truth_cross_val(args: Mapping[str, Any], path_dict: Mapping[str, Any], idx: int) -> Tuple[List[int], ...]:

    '''
    Return ground truth coordinates for the x and y coordinates in the field of view (fov) over the fixed cross validation dataset.

    Args:
        args: non-tunable hyperparameters
        path_dict: dictionary for dataset access with custom file_open function and h5py
        idx: index for indexing the dataset

    Returns:
        path_to_file: path to the data file
        handle: integer that is used as pointer to the file for data access
        cross_val_indices: list of indices indexing the dataset and the ground truth coordinates
        ground_truth_x_fov: list of ground truth x coordinates
        ground_truth_y_fov: list of ground truth y coordinates
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
        
    ground_truth_x_fov = ground_truth_x - offset_x
    ground_truth_y_fov = ground_truth_y - offset_y
    cross_val_dict = pickle.load(open(f'{args.cross_val_dict_path}', 'rb'))
    path_to_cross_val_indices = '{}_{}'.format(path_dict[idx]['user'], path_dict[idx]['date'])
    cross_val_indices = cross_val_dict[path_to_cross_val_indices]
    ground_truth_x_fov = ground_truth_x_fov[cross_val_indices]
    ground_truth_y_fov = ground_truth_y_fov[cross_val_indices]    

    return path_to_file, handle, cross_val_indices, ground_truth_x_fov, ground_truth_y_fov
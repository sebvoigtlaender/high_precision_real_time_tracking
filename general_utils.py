import os, pickle
from typing import Any, List, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import numpy.random as rnd

from tqdm import tqdm

import torch
import torch as pt
import torch.nn.functional as F
import torchvision

from file_sys_utils import *

__all__ = ['ListType',
           'TensorType', 
           'TensorDict', 
           'gaussian_blur',
           'find_baseline_false',
           'find_candidate_coordinates', 
           'find_coordinate_baseline', 
           'get_candidate_coordinates_and_peaks', 
           'find_coordinate_with_prior_no_model', 
           'find_coordinate_with_prior', 
           'get_hp',
           'len_contiguous_idx_blocks',
           'contiguous_idx_blocks',
           'len_block_list',
           'split_dataset',
           'convert_to_hdf']


ListType = Union[List, np.ndarray]
TensorType = Union[pt.Tensor, pt.LongTensor]
TensorDict = MutableMapping[str, TensorType]

gaussian_blur = torchvision.transforms.GaussianBlur(9, 4.0)


def find_baseline_false(idx_list_baseline: List[int],
                        x_difference_baseline: List[int],
                        y_difference_baseline: List[int],
                        pixel_cutoff: Optional[int] = 8) -> List[int]:

    '''
    Return indices in idx_list_baseline where x_difference_baseline or y_difference_baseline are > pixel_cutoff

    Args:
        idx_list_baseline: list of indices included in baseline test
        x_difference_baseline: list of x coordinate differences from ground truth
        y_difference_baseline: list of y coordinate differences from ground truth
        pixel_cutoff: error tolerance threshold

    Returns:
        idx_list_baseline_false: list of indices in idx_list_baseline where x_difference_baseline or y_difference_baseline are > pixel_cutoff
    '''

    assert len(idx_list_baseline) == len(x_difference_baseline)
    idx_false = np.where(np.sqrt(np.array(x_difference_baseline)**2 + np.array(y_difference_baseline)**2) > pixel_cutoff)[0]
    
    idx_list_baseline_false = list(np.array(idx_list_baseline)[idx_false])
    return idx_list_baseline_false


def find_candidate_coordinates(x: TensorType) -> Tuple[List[int], ...]:
    '''
    Find all local maxima in heatmap

    Args:
        x: network output heatmap

    Returns: 
        candidate_x_coordinates: list of x coordinates of all detected local maxima
        candidate_y_coordinates: list of y coordinates of all detected local maxima

    Example:
    input =
        tensor([[[[0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 1.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 1.0000],
              [0.0000, 0.0000, 0.6000, 0.0000]]]])
    xput:
        ([1, 3], [1, 2])
    '''
    peak_indicator = (F.max_pool2d(x, kernel_size=1, stride=1, padding=0, return_indices=True)[1]
                   == F.max_pool2d(x, kernel_size=3, stride=1, padding=1, return_indices=True)[1])
    peak_indicator[:, :, 0:2, 0:2] = peak_indicator[:, :, 0:2, 0:2] * x[:, :, 0:2, 0:2]
    candidate_x_coordinates = (peak_indicator == True).nonzero(as_tuple=True)[3]
    candidate_y_coordinates = (peak_indicator == True).nonzero(as_tuple=True)[2]
    candidate_x_coordinates = candidate_x_coordinates.cpu().detach().numpy()
    candidate_y_coordinates = candidate_y_coordinates.cpu().detach().numpy()
    return candidate_x_coordinates, candidate_y_coordinates


def find_coordinate_baseline(x: TensorType,
                             high_pass_filter: Optional[float] = 3e-2) -> Tuple[int, ...]:

    '''
    Find coordinates by picking the maximum

    Args:
        x: network output heatmap
        high_pass_filter: high pass filter for filtering local maxima in noise,
        drastically reduces the computational burden on find_candidate_coordinates(x)

    Returns: 
        posterior_x: x coordinate at current frame
        posterior_y: y coordinate at current frame
    '''
    
    if x.max().item() < high_pass_filter:
        x[x < x.max().item()/2] = 0
    else:
        x[x < high_pass_filter] = 0
        
    candidate_coordinates = (x == x.max().item()).nonzero(as_tuple=True)
    
    if not len(candidate_coordinates[3]) == 1:
        posterior_x = candidate_coordinates[3][0].item()
    else:
        posterior_x = candidate_coordinates[3].item()
        
    if not len(candidate_coordinates[2]) == 1:
        posterior_y = candidate_coordinates[2][0].item()
    else:
        posterior_y = candidate_coordinates[2].item()
            
    return posterior_x, posterior_y


def get_candidate_coordinates_and_peaks(x: TensorType,
                                        high_pass_filter: Optional[float] = 3e-2) -> Tuple[List[int], List[int], List[float]]:

    if x.max().item() < high_pass_filter:
        x[x < x.max().item()/2] = 0
    else:
        x[x < high_pass_filter] = 0

    candidate_x_coordinates, candidate_y_coordinates = find_candidate_coordinates(x)
    peak_values = np.array([x[0, 0].cpu().detach().numpy()[candidate_y_coordinates[i], candidate_x_coordinates[i]] for i in range(len(candidate_x_coordinates))])
    return candidate_x_coordinates, candidate_y_coordinates, peak_values


def find_coordinate_with_prior_no_model(candidate_x_coordinates: List[int],
                                        candidate_y_coordinates: List[int],
                                        peak_values: List[float],
                                        prior_x: int,
                                        prior_y: int,
                                        i: int, 
                                        high_pass_filter: float, 
                                        correlation_scale: float,
                                        hard_boundary: int) -> Tuple[int, ...]:

    '''
    Find coordinate by:
    1. Evaluating the absolute distances of the candidate coordinates from the prior
    2. Weighting the grayscale values at the discovered candidate coordinate by their absolute distance from the prior
    3. Updating prior, if the new coordinate is not beyond a biologically plausible boundary value.
       In that case, avoid hard failure by not updating prior (soft failure)

    Args:
        candidate_x_coordinates: all candidate x coordinates found in heatmap
        candidate_y_coordinates: all candidate y coordinates found in heatmap
        peak_values: all peak values found in heatmap
        prior_x: x coordinate at frame before the current frame x, treated as prior
        prior_y: y coordinate at frame before the current frame x, treated as prior
        i: index of current frame
        high_pass_filter: high pass filter for filtering local maxima in noise,
        drastically reduces the computational burden on find_candidate_coordinates(x)
        correlation_scale: scale that determines how strongly a distance to the prior coordinate is weighted
        hard_boundary: boundary value that determines whether the prior is updated depending on the distance between
        the best candidate coordinate and the prior

    Returns: 
        posterior_x: x coordinate at current frame
        posterior_y: y coordinate at current frame
    '''
    
    if i == 0:
        posterior_x = prior_x
        posterior_y = prior_y

    if not i == 0: # no prior on the first frame

        radial_coordinate_distances = [np.sqrt((candidate_x_coordinates[i] - prior_x)**2 + (candidate_y_coordinates[i] - prior_y)**2) for i in range(len(candidate_x_coordinates))]
        weighted_peaks = [np.exp(-correlation_scale*distance) * peak_value for distance, peak_value in zip(radial_coordinate_distances, peak_values)]
        opt_idx = np.where(weighted_peaks == np.max(weighted_peaks))[0][0]

        if np.abs(candidate_x_coordinates[opt_idx] - prior_x) + np.abs(candidate_y_coordinates[opt_idx] - prior_y) < hard_boundary:
            posterior_x = candidate_x_coordinates[opt_idx]
            posterior_y = candidate_y_coordinates[opt_idx]
        else:
            posterior_x = prior_x
            posterior_y = prior_y
    
    return posterior_x, posterior_y


def find_coordinate_with_prior(x: TensorType,
                               prior_x: int,
                               prior_y: int,
                               i: int, 
                               high_pass_filter: float, 
                               correlation_scale: float,
                               hard_boundary: int) -> Tuple[int, ...]:

    '''
    Find coordinate by:
    1. Thresholding heatmap with high pass filter to avoid evaluating local maxima in noise background
    2. Finding all candidate coordinates by searching the heatmap for maxima
    3. Evaluating the absolute distances of the candidate coordinates from the prior
    4. Weighting the grayscale values at the discovered candidate coordinate by their absolute distance from the prior
    5. Updating prior, if the new coordinate is not beyond a biologically plausible boundary value.
       In that case, avoid hard failure by not updating prior (soft failure)

    Args:
        x: network output heatmap
        prior_x: x coordinate at frame before the current frame x, treated as prior
        prior_y: y coordinate at frame before the current frame x, treated as prior
        i: index of current frame
        high_pass_filter: high pass filter for filtering local maxima in noise,
        drastically reduces the computational burden on find_candidate_coordinates(x)
        correlation_scale: scale that determines how strongly a distance to the prior coordinate is weighted
        hard_boundary: boundary value that determines whether the prior is updated depending on the distance between
        the best candidate coordinate and the prior

    Returns: 
        posterior_x: x coordinate at current frame
        posterior_y: y coordinate at current frame
    '''
    
    if i == 0:
        posterior_x, posterior_y = find_coordinate_baseline(x, high_pass_filter)

    if not i == 0: # no prior on the first frame

        if x.max().item() < high_pass_filter: # implements high pass filter for noise
            x[x < x.max().item()/2] = 0
        else:
            x[x < high_pass_filter] = 0

        candidate_x_coordinates, candidate_y_coordinates = find_candidate_coordinates(x)
        peak_values = [x[0, 0].cpu().detach().numpy()[candidate_y_coordinates[i], candidate_x_coordinates[i]] for i in range(len(candidate_x_coordinates))]

        radial_coordinate_distances = [np.sqrt((candidate_x_coordinates[i] - prior_x)**2 + (candidate_y_coordinates[i] - prior_y)**2) for i in range(len(candidate_x_coordinates))]
        weighted_peaks = [np.exp(-correlation_scale*distance) * peak_value for distance, peak_value in zip(radial_coordinate_distances, peak_values)]
        opt_idx = np.where(weighted_peaks == np.max(weighted_peaks))[0][0]

        if np.abs(candidate_x_coordinates[opt_idx] - prior_x) + np.abs(candidate_y_coordinates[opt_idx] - prior_y) < hard_boundary:
            posterior_x = candidate_x_coordinates[opt_idx]
            posterior_y = candidate_y_coordinates[opt_idx]
        else:
            posterior_x = prior_x
            posterior_y = prior_y
    
    return posterior_x, posterior_y


def get_hp(path_hp: Optional[str] = ''):

    correlation_scale, hard_boundary = 0.0, 512
    if os.path.exists(path_hp):
        hp_list = pickle.load(open(path_hp, 'rb'))
        correlation_scale, hard_boundary = hp_list
    return correlation_scale, hard_boundary


def len_contiguous_idx_blocks(idx_list: List[int]) -> Tuple[List[int], ...]:
    '''
    Return length of contiguous blocks in list of indices idx_list

    Args:
        idx_list: list of indices

    Returns: 
        len_block_list: list of block lengths

    Example: 
    input = [1, 2, 3, 5, 6]
    output = [3, 2]
    '''

    start_idx = idx_list[0]
    stop_idx = idx_list[0]
    len_block_list = []

    for idx in idx_list[1:]:
        if idx == stop_idx + 1:
            stop_idx = idx
            continue
        elif not idx == stop_idx + 1:
            len_block_list.append(stop_idx-start_idx+1)
            start_idx = idx
            stop_idx = idx
    if stop_idx == idx_list[-1]:
        len_block_list.append(stop_idx-start_idx+1)
    return len_block_list


def contiguous_idx_blocks(idx_list: List[int],
                          min_len: Optional[int] = 100) -> List[List[int]]:

    '''
    Return list of lists of contiguous index blocks of minimum length
    specified by min_len.
    Keep in mind that we keep the first index of each continuous index
    block - when calling cross_validate() the ground truth is indexed
    by the first index of each block. cross_validate() accounts for
    the possibility of an empty list.

    Args:
        idx_list: list of indices
        min_len: minimum length of contiguous index block, if len(block) < min_len it is excluded
        from the final list.

    Returns: 
        block_list: list of lists of contiguous indices 

    Example: 
    min_len = 3
    input = [1, 2, 3, 5, 6, 8, 9, 10]
    output = [[1, 2, 3], [8, 9, 10]]
    '''

    start_idx = idx_list[0]
    stop_idx = idx_list[0]
    block_list = []
    idx_block = []

    for idx in idx_list[1:]:
        if idx == stop_idx + 1:
            stop_idx = idx
            if not start_idx == idx:
                idx_block.append(idx)
            continue
        elif not idx == stop_idx + 1:
            len_block = stop_idx - start_idx + 1
            idx_block.insert(0, start_idx)
            if len_block >= min_len:
                block_list.append(idx_block)
            start_idx = idx
            stop_idx = idx
            idx_block = []
    if stop_idx == idx_list[-1]:
        len_block = stop_idx - start_idx + 1
        if len_block >= min_len:
            idx_block.insert(0, start_idx)
            block_list.append(idx_block)
    return block_list


def len_block_list(block_list: List[int]) -> int:
    '''
    Return total number of indices in list of index blocks
    '''
    n = 0
    for idx_block in block_list:
        n += len(idx_block)
    return n


def split_dataset(args: Mapping[str, Any],
                  indices: List[int], k: int) -> Tuple[List[int], ...]:
    '''
    Split indices of dataset in train and evaluation dataset indices
    '''
    assert args.k > k
    eval_indices = indices[(len(indices)//args.k)*k:(len(indices)//args.k)*(k + 1)]
    train_indices = list(set(indices) - set(eval_indices))
    return train_indices, eval_indices


def convert_to_hdf(args: Mapping[str, Any]) -> None:

    path_dict = get_path_dict(f'{args.test_dataset_id}')
    for idx in range(len(path_dict)):
        test_result_path = '{}_{}_{}'.format(args.test_result_path, path_dict[idx]['user'], path_dict[idx]['date'])
        result_list = pickle.load(open(f'{test_result_path}_final.pkl', 'rb'))
        coordinates = np.stack(result_list)
        with h5py.File(f'{test_result_path}_dnn_tracking.hdf5', 'w') as f:
            f.create_dataset('coordinates', data=coordinates)
            f.close()
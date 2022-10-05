from typing import Any, Mapping
import numpy as np

from config import get_path_dict
from data_utils import get_x, get_ground_truth, get_ground_truth_cross_val
from file_sys_utils import file_read, file_close
from general_utils import contiguous_idx_blocks, find_coordinate_baseline, split_dataset


def cross_validate_k_fold(args: Mapping[str, Any],
                          path_dict: Mapping[str, Any],
                          model: Any,
                          k: int,
                          len_cross_val: int = int(10e6),
                          high_pass_filter: float = 3e-2,
                          correlation_scale: float = 0.0,
                          hard_boundary: int = 512) -> Mapping[str, Any]:  
    '''
    k fold cross validation

    Args:
        args: non-tunable hyperparameter
        path_dict: dictionary for dataset access
        model: torch model
        k: k-fold cross validation parameter
        len_cross_val: maximum length of each index block to be evaluated until proceeding to the next index block
        high_pass_filter: high pass filter for filtering local maxima in noise,
        drastically reduces the computational burden on find_candidate_coordinates(x)
        correlation_scale: scale that determines how strongly a distance to the prior coordinate is weighted
        hard_boundary: boundary value that determines whether the prior is updated depending on the distance between
        the best candidate coordinate and the prior

    Returns:
        cross_val_ratio: number of frames in cross validation dataset with radial distance between heatmap maximum and ground truth
        below error tolerance threshold args.pixel_cutoff
    '''

    model.eval()
    x_diff = []
    y_diff = []
    
    for idx in range(len(path_dict)):
        
        filename, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)
        train_indices, eval_indices = split_dataset(args, indices, k)
        eval_indices = contiguous_idx_blocks(eval_indices, args.min_len_block)

        for idx_block in eval_indices:
            for i in idx_block[0:len_cross_val]:
                x = get_x(handle, i)
                out = model.infer(x).to(args.device)
                x_coordinate, y_coordinate = find_coordinate_baseline(out, high_pass_filter)
                x_diff.append(x_coordinate - ground_truth_x_fov[i])
                y_diff.append(y_coordinate - ground_truth_y_fov[i])

        file_close(handle)
    result_list = np.sqrt(np.array(x_diff)**2 + np.array(y_diff)**2)
    cross_val_ratio = len(result_list[result_list < args.pixel_cutoff])/len(result_list)
    model.train()
    return cross_val_ratio


def cross_validate(args: Mapping[str, Any],
                   model: Any) -> Mapping[str, Any]:  
    '''
    Cross validate on fixed cross validation dataset

    Args:
        args: non-tunable hyperparameters
        model: torch model

    Returns:
        cross_val_ratio: number of frames in cross validation dataset with radial distance between heatmap maximum and ground truth
        below error tolerance threshold args.pixel_cutoff
    '''

    model.eval()
    x_diff = []
    y_diff = []

    path_dict = get_path_dict(args.cross_val_dataset_id)
    
    for idx in range(len(path_dict)):
        filename, handle, cross_val_indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth_cross_val(args, path_dict, idx)
        for i, j in enumerate(cross_val_indices):
            x = get_x(handle, j)
            out = model.infer(x).to(args.device)
            x_coordinate, y_coordinate = find_coordinate_baseline(out, args.high_pass_filter)
            x_diff.append(x_coordinate - ground_truth_x_fov[i])
            y_diff.append(y_coordinate - ground_truth_y_fov[i])
        file_close(handle)

    result_list = np.sqrt(np.array(x_diff)**2 + np.array(y_diff)**2)
    cross_val_ratio = len(result_list[result_list < args.pixel_cutoff])/len(result_list)
    model.train()
    return cross_val_ratio
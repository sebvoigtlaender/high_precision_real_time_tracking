import os, pickle
from typing import Any, Mapping, Optional

from absl import logging
from tqdm import tqdm

import numpy as np
import torch as pt

from ops.arguments import get_args
from ops.config import get_config, get_path_dict
from ops.file_sys_utils import file_close, get_path_to_result, get_path_to_file
from ops.data_utils import get_x, get_ground_truth
from ops.general_utils import find_coordinate_baseline, find_coordinate_with_prior, find_coordinate_with_prior_no_model, get_candidate_coordinates_and_peaks, contiguous_idx_blocks, get_hp
from ops.model import Model


def eval_baseline(args: Mapping[str, Any],
                  path_dict: Mapping[str, Any],
                  model: Any,
                  correlation_scale: Optional[float] = 0.0,
                  hard_boundary: Optional[int] = 512) -> Mapping[str, Any]:

    '''
    Test trained or optimized model on unseen data
    '''

    model.eval()
    for idx in range(len(path_dict)):

        filename, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)
        test_indices = contiguous_idx_blocks(indices, args.min_len_block)

        idx_list = []
        x_difference_prior = []
        y_difference_prior = []
        candidate_x_coordinate_list = []
        candidate_y_coordinate_list = []
        peak_list = []

        for idx_block in test_indices:
            
            i = idx_block[0]
            x = get_x(handle, i)
            out = model.infer(x).to(args.device)
            x_coordinate, y_coordinate = find_coordinate_baseline(out, args.high_pass_filter)

            for i in tqdm(idx_block[1:args.max_len_block+1]):
                x = get_x(handle, i)
                out = model.infer(x).to(args.device)
                candidate_x_coordinates, candidate_y_coordinates, peak_values = get_candidate_coordinates_and_peaks(out, args.high_pass_filter)
                
                if args.test_baseline:
                    x_coordinate, y_coordinate = find_coordinate_baseline(out, args.high_pass_filter)
                else:
                    x_coordinate, x_coordinate = find_coordinate_with_prior(out, x_coordinate, y_coordinate, i, args.high_pass_filter, correlation_scale, hard_boundary)

                idx_list.append(i)
                x_difference_prior.append(x_coordinate - ground_truth_x_fov[i])
                y_difference_prior.append(y_coordinate - ground_truth_y_fov[i])
                candidate_x_coordinate_list.append(candidate_x_coordinates)
                candidate_y_coordinate_list.append(candidate_y_coordinates)
                peak_list.append(peak_values)
            
        file_close(handle)
        
        result_list = [idx_list, x_difference_prior, y_difference_prior, candidate_x_coordinate_list, candidate_y_coordinate_list, peak_list]
        path_to_result = get_path_to_result(path_dict, idx)
        pickle.dump(result_list, open(path_to_result, 'wb'))


def eval_opt(args: Mapping[str, Any],
             path_dict: str,
             correlation_scale: float,
             hard_boundary: int) -> Mapping[str, Any]:
    
    '''
    Evaluate trained model with optized hyperparameters on unseen data based on baseline test run. No images or model inference is needed.
    '''
 
    for idx in range(len(path_dict)):
    
        path_to_file = get_path_to_file(path_dict, idx)
        path_to_result = get_path_to_result(path_dict, idx)
        correlation_scale, hard_boundary = get_hp(args.hp)

        x_coordinate_list = []
        y_coordinate_list = []

        test_result_list = pickle.load(open(f'{path_to_result}', 'rb'))
        idx_list_baseline, x_difference_baseline, y_difference_baseline, candidate_x_coordinate_list, candidate_y_coordinate_list, peak_list = test_result_list
        ground_truth_path_to_file, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)
        val_indices = contiguous_idx_blocks(indices, min_len = args.min_len_block)

        assert path_to_file == ground_truth_path_to_file

        idx_list_baseline = iter(idx_list_baseline)
        candidate_x_coordinate_list = iter(candidate_x_coordinate_list)
        candidate_y_coordinate_list = iter(candidate_y_coordinate_list)
        peak_list = iter(peak_list)

        for idx_block in val_indices:

            x_coordinate, y_coordinate = ground_truth_x_fov[idx_block[0]], ground_truth_y_fov[idx_block[0]]
            for i in tqdm(idx_block[1:args.max_len_block+1]):

                i_baseline = next(idx_list_baseline)
                assert i == i_baseline
                candidate_x_coordinates = next(candidate_x_coordinate_list)
                candidate_y_coordinates = next(candidate_y_coordinate_list)
                peak_values = next(peak_list)

                x_coordinate, y_coordinate = find_coordinate_with_prior_no_model(candidate_x_coordinates,
                                                                               candidate_y_coordinates,
                                                                               peak_values,
                                                                               x_coordinate,
                                                                               y_coordinate,
                                                                               i,
                                                                               args.high_pass_filter,
                                                                               correlation_scale,
                                                                               hard_boundary)


                x_coordinate_list.append(x_coordinate)
                y_coordinate_list.append(y_coordinate)

        coordinates = np.stack([x_coordinate_list, y_coordinate_list])
        path_to_result = get_path_to_result(path_dict, idx)
        path_to_result = f'{path_to_result}_hp'
        pickle.dump(coordinates, open(path_to_result, 'wb'))


def main():

    logging.set_verbosity(logging.INFO)
    args = get_args()
    args = get_config(args)

    args.device = pt.device(f'cuda:{args.device_idx}' if pt.cuda.is_available() else 'cpu')
    path_dict = get_path_dict(f'{args.eval_dataset_id}')
    
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.eval_type == 'model-based':
        state_dict_path = f'{args.state_dict_path}'
        model = Model(args, core_type = args.core_type).to(args.device)
        model.load_state_dict(pt.load(state_dict_path, map_location=f'{args.device}'))
        eval_baseline(args, path_dict, model)
        
    elif args.eval_type == 'model-free':
        correlation_scale, hard_boundary = get_hp(args.hp)
        eval_opt(args, path_dict, correlation_scale, hard_boundary)

        
if __name__  == '__main__':
    main()
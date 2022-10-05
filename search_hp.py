import pickle 
from typing import Any, Mapping

from absl import logging
from tqdm import tqdm

import numpy.random as rnd
import torch as pt

from ops.arguments import get_args
from ops.config import get_config, get_path_dict
from ops.file_sys_utils import file_close, get_path_to_result, get_path_to_file
from ops.data_utils import get_x, get_ground_truth
from ops.general_utils import find_coordinate_baseline, find_coordinate_with_prior, find_coordinate_with_prior_no_model, contiguous_idx_blocks
from ops.model import Model


def search_hp(args: Mapping[str, Any],
               path_dict: Mapping[str, Any],
               model: Any,
               n_epochs_opt_hyperp: int,
               n_frames_opt_hyperp: int) -> Mapping[str, Any]:
    '''
    Optimize post-training hyperparameters for error correction
    '''
    model.eval()
    result_dict = {}

    for t in range(n_epochs_opt_hyperp):

        correlation_scale = rnd.choice(args.correlation_scale)
        hard_boundary = rnd.choice(args.hard_boundary)
        
        logging.info((correlation_scale, hard_boundary))

        counter = 1

        for idx in range(len(path_dict)):
            
            if counter > n_frames_opt_hyperp:
                break
            
            x_difference_prior = []
            y_difference_prior = []
            path_to_file, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)

            val_indices = contiguous_idx_blocks(indices, args.min_len_block)

            for idx_block in val_indices:
                
                if counter > n_frames_opt_hyperp:
                    break
                    
                i = idx_block[0]
                x = get_x(handle, i)
                out = model.infer(x).to(args.device)
                x_coordinate, y_coordinate = find_coordinate_baseline(out, args.high_pass_filter)
                
                for i in tqdm(idx_block[1:n_frames_opt_hyperp+1]):
                    if counter > n_frames_opt_hyperp:
                        break
                    x = get_x(handle, i)
                    out = model.infer(x).to(args.device)
                    x_coordinate, y_coordinate = find_coordinate_with_prior(out, x_coordinate, y_coordinate, i, args.high_pass_filter, correlation_scale, hard_boundary)
                    counter += 1

                    x_difference_prior.append(x_coordinate - ground_truth_x_fov[i])
                    y_difference_prior.append(y_coordinate - ground_truth_y_fov[i])

            file_close(handle)
            path_to_result = get_path_to_result(path_dict, idx)
            result_dict[f'{path_to_result}'] = [x_difference_prior, y_difference_prior]

        result_dict['hp'] = (correlation_scale, hard_boundary)
        pickle.dump(result_dict, open(f'{args.search_result_path}_{t}', 'wb'))
        result_dict = {}


def search_hp_model_free(args: Mapping[str, Any],
               path_dict: Mapping[str, Any],
               n_epochs_opt_hyperp: int) -> Mapping[str, Any]:
    '''
    Optimize post-training hyperparameters for error correction
    '''
    for t in range(n_epochs_opt_hyperp):
        
        result_dict = {}

        correlation_scale = rnd.choice(args.correlation_scale)
        hard_boundary = rnd.choice(args.hard_boundary)

        logging.info((correlation_scale, hard_boundary))

        for idx in range(len(path_dict)):

            x_difference_prior = []
            y_difference_prior = []
            
            path_to_file = get_path_to_file(path_dict, idx)
            path_to_result = get_path_to_result(path_dict, idx)
            test_result_list = pickle.load(open(path_to_result, 'rb'))
            idx_list_baseline, x_difference_baseline, y_difference_baseline, candidate_x_coordinate_list, candidate_y_coordinate_list, peak_list = test_result_list
            ground_truth_path_to_file, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)
            val_indices = contiguous_idx_blocks(indices, args.min_len_block)

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


                    x_difference_prior.append(x_coordinate - ground_truth_x_fov[i])
                    y_difference_prior.append(y_coordinate - ground_truth_y_fov[i])

            result_dict[f'{path_to_result}'] = [x_difference_prior, y_difference_prior]

        result_dict['hp'] = (correlation_scale, hard_boundary)
        pickle.dump(result_dict, open(f'{args.search_result_path}_{t}', 'wb'))
        result_dict = {}


def main():

    logging.set_verbosity(logging.INFO)
    args = get_args()
    args = get_config(args)
    args.device = pt.device(f'cuda:{args.device_idx}' if pt.cuda.is_available() else 'cpu')
    path_dict = get_path_dict(f'{args.search_hp_dataset_id}')
    
    if args.search_hp_type == 'model-based':
        model = Model(args, core_type = args.core_type).to(args.device)
        state_dict_path = f'{args.state_dict_path}'
        model.load_state_dict(pt.load(state_dict_path, map_location=f'{args.device}'))
        search_hp(args, path_dict, model, args.n_epochs_opt_hyperp, args.n_frames_opt_hyperp)
    elif args.search_hp_type == 'model-free':
        assert args.eval_dataset_id == args.search_hp_dataset_id
        search_hp_model_free(args, path_dict, args.n_epochs_opt_hyperp)

        
if __name__  == '__main__':
    main()
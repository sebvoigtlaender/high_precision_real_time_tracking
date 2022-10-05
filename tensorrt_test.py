import os, pickle, h5py, sys
from typing import Any, Mapping

from absl import logging
from tqdm import tqdm

import numpy as np
import torch as pt

from ops.arguments import get_args
from ops.config import get_config, get_path_dict
from ops.file_sys_utils import file_close, file_get_n_frames, get_path_to_result
from ops.data_utils import get_x, get_ground_truth
from ops.general_utils import find_coordinate_baseline, find_coordinate_with_prior, get_hp
from tensorrt.tensorrt_model import Model
from ops.tracking_utils import get_x_tracking, get_ground_truth_tracking, update_anchor_coordinates

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)


def test(args: Mapping[str, Any],
                  path_dict: Mapping[str, Any],
                  model: Any,
                  correlation_scale: float, 
                  hard_boundary: int):

    '''
    Test model on unseen data
    '''

    for idx in range(len(path_dict)):

        filename, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)
        n_test_indices = file_get_n_frames(handle)
        x_coordinate_list = []
        y_coordinate_list = []

        for i in tqdm(range(n_test_indices)):
            x = get_x(handle, i)
            out = model.infer(x).to(args.device)
            pt.cuda.synchronize()

            if i == 0:
                x_coordinate, y_coordinate = find_coordinate_baseline(out, args.high_pass_filter)
            elif i > 0:
                x_coordinate, y_coordinate = find_coordinate_with_prior(out, x_coordinate, y_coordinate, i, args.high_pass_filter, correlation_scale, hard_boundary)
            
            x_coordinate_list.append(x_coordinate)
            y_coordinate_list.append(y_coordinate)

        file_close(handle)
        coordinates = np.stack([x_coordinate_list, y_coordinate_list])
        path_to_result = get_path_to_result(path_dict, idx)
        pickle.dump(coordinates, open(f'{path_to_result}_hp', 'wb'))


def track(args: Mapping[str, Any],
          path_dict: Mapping[str, Any],
          model: Any,
          correlation_scale: float,
          hard_boundary: int):

    '''
    Simulated live tracking

    Args:
        args: non-tunable hyperparameters
        path_dict: dictionary of paths for opening datasets
        model: deep neural network
        correlation_scale, hard_boundary: tunable hyperparameters
    '''

    for idx in range(len(path_dict)):

        path_to_file, handle, ground_truth_x, ground_truth_y, offset_x, offset_y, x_background = get_ground_truth_tracking(path_dict, idx)

        n_test_indices = file_get_n_frames(handle)
        x_coordinate_list = []
        y_coordinate_list = []
        x_anchor_list = []
        y_anchor_list = []
        diff_ground_truth_x = []
        diff_ground_truth_y = []

        for i in tqdm(range(n_test_indices)):
            if i == 0:
                x_anchor = offset_x[i]
                y_anchor = offset_y[i]
                x = get_x(handle, i)
                out = model.infer(x).to(args.device)
                pt.cuda.synchronize()
                x_coordinate, y_coordinate = find_coordinate_baseline(out, args.high_pass_filter)
            elif i > 0:
                x_anchor, y_anchor = update_anchor_coordinates(args, x_anchor, y_anchor, x_coordinate, y_coordinate)
                x = get_x_tracking(args, handle, x_background, offset_x, offset_y, x_anchor, y_anchor, i)
                out = model.infer(x).to(args.device)
                pt.cuda.synchronize()
                x_coordinate, y_coordinate = find_coordinate_with_prior(out, x_coordinate, y_coordinate, i, args.high_pass_filter, correlation_scale, hard_boundary)

            translated_ground_truth_x_fov = ground_truth_x[i] - x_anchor
            translated_ground_truth_y_fov = ground_truth_y[i] - y_anchor
            
            x_coordinate_list.append(x_coordinate)
            y_coordinate_list.append(y_coordinate)
            x_anchor_list.append(x_anchor)
            y_anchor_list.append(y_anchor)
            diff_ground_truth_x.append(translated_ground_truth_x_fov - x_coordinate)
            diff_ground_truth_y.append(translated_ground_truth_y_fov - y_coordinate)
            if np.sqrt(diff_ground_truth_x[-1]**2 + diff_ground_truth_y[-1]**2) > args.h:
                logging.info(f'specimen lost after {i} frames')
                break

        coordinates = np.stack([x_coordinate_list, y_coordinate_list])
        
        anchor_list = np.stack([x_anchor_list, y_anchor_list])
        radial_distance_ground_truth = np.sqrt(np.array(diff_ground_truth_x)**2 + np.array(diff_ground_truth_y)**2)

        path_to_result = get_path_to_result(path_dict, idx)
        with h5py.File(f'{path_to_result}_track.hdf5', 'w') as f:
            f.create_dataset('coordinates', data=coordinates)
            f.create_dataset('anchor_list', data=anchor_list)
            f.create_dataset('radial_distance_ground_truth', data=radial_distance_ground_truth)
            f.close()      

        file_close(handle)


def main():

    logging.set_verbosity(logging.INFO)
    args = get_args()
    args = get_config(args)

    args.device = pt.device(f'cuda:{args.device_idx}' if pt.cuda.is_available() else 'cpu')
    logging.info('args.precision = {args.precision}')
    path_dict = get_path_dict(f'{args.track_dataset_id}')
    
    if not os.path.exists('results'):
        os.makedirs('results')

    state_dict_path = f'{args.state_dict_path}'
    model = Model(args).to(args.device)
    model.load_state_dict(pt.load(state_dict_path, map_location=f'{args.device}'), strict=False)
    correlation_scale, hard_boundary = get_hp(args.hp)
    
    logging.info(f'correlation_scale = {correlation_scale}, boundary = {hard_boundary}')
    if args.test_type == 'test':
        test(args, path_dict, model, correlation_scale, hard_boundary)
    elif args.test_type == 'track':
        track(args, path_dict, model, correlation_scale, hard_boundary)


if __name__  == '__main__':
    main()
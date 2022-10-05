import os, h5py, sys
from typing import Any, Mapping

from absl import logging

import numpy as np
import torch as pt

from ops.arguments import get_args
from ops.config import get_config
from ops.file_sys_utils import file_close, get_path_to_result
from ops.general_utils import find_coordinate_baseline, find_coordinate_with_prior, get_hp
from ops.model import Model
from ops.tracking_utils import update_anchor_coordinates

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)


def track(args: Mapping[str, Any],
          model: Any,
          correlation_scale: float,
          hard_boundary: int):

    '''
    Live tracking template. The user needs to specify
        – a stopping criterion for the loop, e.g., the lenght of the experiment or a criterion for losing the specimen,
        – how to get the first anchor coordinates
        – a get_x function for loading the frame

    Args:
        args: non-tunable hyperparameters
        model: deep neural network
        correlation_scale, hard_boundary: tunable hyperparameters
    '''

    for idx in range(len(path_dict)):

        x_coordinate_list = []
        y_coordinate_list = []
        x_anchor_list = []
        y_anchor_list = []
        i = 0

        # while : 

            if i == 0:
                # x_anchor = 
                # y_anchor = 
                # x = get_x()
                out = model.infer(x).to(args.device)
                x_coordinate, y_coordinate = find_coordinate_baseline(out, args.high_pass_filter)
            elif i > 0:
                x_anchor, y_anchor = update_anchor_coordinates(args, x_anchor, y_anchor, x_coordinate, y_coordinate)
                # x = get_x()
                out = model.infer(x).to(args.device)
                x_coordinate, y_coordinate = find_coordinate_with_prior(out, x_coordinate, y_coordinate, i, args.high_pass_filter, correlation_scale, hard_boundary)

            i += 1
            
            x_coordinate_list.append(x_coordinate)
            y_coordinate_list.append(y_coordinate)
            x_anchor_list.append(x_anchor)
            y_anchor_list.append(y_anchor)

        coordinates = np.stack([x_coordinate_list, y_coordinate_list])
        anchor_list = np.stack([x_anchor_list, y_anchor_list])

        path_to_result = get_path_to_result(path_dict, idx)
        with h5py.File(f'{path_to_result}_track.hdf5', 'w') as f:
            f.create_dataset('coordinates', data=coordinates)
            f.create_dataset('anchor_list', data=anchor_list)
            f.close()      

        file_close(handle)


def main():

    logging.set_verbosity(logging.INFO)
    args = get_args()
    args = get_config(args)

    args.device = pt.device(f'cuda:{args.device_idx}' if pt.cuda.is_available() else 'cpu')
    
    if not os.path.exists('results'):
        os.makedirs('results')

    state_dict_path = f'{args.state_dict_path}'
    model = Model(args, core_type = args.core_type).to(args.device)
    model.load_state_dict(pt.load(state_dict_path, map_location=f'{args.device}'), strict=False)
    correlation_scale, hard_boundary = get_hp(args.hp)
    logging.info(f'correlation_scale = {correlation_scale}, boundary = {hard_boundary}')
    track(args, model, correlation_scale, hard_boundary)


if __name__  == '__main__':
    main()
import pickle
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import numpy.random as rnd
import torch as pt

from data_utils import get_x, get_target, get_ground_truth
from file_sys_utils import file_close, file_read, get_path_to_result
from general_utils import ListType, TensorType, find_baseline_false, find_coordinate_baseline, contiguous_idx_blocks, split_dataset


class Data():

    def __init__(self, args: Mapping[str, Any]) -> None:
        self.args = args
        self.device = args.device

    def load(self, input_frames: ListType, targets: ListType) -> Tuple[TensorType]:
        while True:
            idx = rnd.choice(range(len(input_frames)), self.args.batch_size, replace = False)
            x = input_frames[idx]
            x = pt.Tensor(x.astype(np.float32)).to(self.device)
            reconstruction_target = targets[idx]
            reconstruction_target = pt.Tensor(reconstruction_target.astype(np.float32)).unsqueeze(1).to(self.device)
            yield x, reconstruction_target


def reject(args: Mapping[str, Any],
                    model: Any,
                    x: ListType,
                    ground_truth_x_fov: ListType,
                    ground_truth_y_fov: ListType,
                    i: int,
                    high_pass_filter: Optional[float] = 3e-2) -> bool:
    '''
    Evaluate model for prioritized batching
    '''
    model.eval()
    prior_x, prior_y = ground_truth_x_fov[i-1], ground_truth_y_fov[i-1]
    with pt.no_grad():
        out = model.infer(x)
    posterior_x, posterior_y = find_coordinate_baseline(out, high_pass_filter)
    x_difference = posterior_x - ground_truth_x_fov[i]
    y_difference = posterior_y - ground_truth_y_fov[i]
    model.train()
    if np.sqrt(x_difference**2 + y_difference**2) > args.pixel_cutoff:
        return False
    return True


def load_dataset(args: Mapping[str, Any],
                 path_dict: Mapping[str, Any],
                 model: Any,
                 k: int,
                 input_frames: TensorType,
                 targets: TensorType,
                 high_pass_filter: Optional[float] = 3e-2) -> Tuple[TensorType, TensorType]:

    assert input_frames.shape == (args.len_block*len(path_dict), 1, 512, 512)
    assert targets.shape == (args.len_block*len(path_dict), 512, 512)
    # print(args.pixel_cutoff)
    
    for idx in range(len(path_dict)):

        path_to_file, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)

        if args.fine_tune:
            path_to_result = get_path_to_result(path_dict, idx)
            result_list = pickle.load(open(f'{path_to_result}', 'rb'))
            idx_list_baseline, x_difference_baseline, y_difference_baseline, candidate_x_coordinate_list, candidate_y_coordinate_list, peak_list = result_list
            train_indices = find_baseline_false(idx_list_baseline, x_difference_baseline, y_difference_baseline, args.pixel_cutoff)
            assert train_indices, 'no hard frames' 

        else:
            train_indices, eval_indices = split_dataset(args, indices, k)

        n = 0
        n_reject = 0

        while n < args.len_block:
            
            i = rnd.choice(train_indices)
            x = get_x(handle, i)
            if reject(args, model, x, ground_truth_x_fov, ground_truth_x_fov, i, high_pass_filter) and n_reject < args.max_n_reject:
                n_reject += 1
                continue
            else:
                target = get_target(args, int(ground_truth_x_fov[i]), int(ground_truth_y_fov[i]))
                input_frames[n + idx*args.len_block, 0] = x
                targets[n + idx*args.len_block] = target
                n += 1
                n_reject = 0
                
        file_close(handle)
        
    return input_frames, targets


def load_dataset_online(args: Mapping[str, Any],
                 path_dict: Mapping[str, Any],
                 model: Any,
                 input_frames: TensorType,
                 targets: TensorType,
                 high_pass_filter: Optional[float] = 3e-2) -> Tuple[TensorType, TensorType]:
    
    n = 0
    idx = rnd.randint(0, len(path_dict))
    path_to_file, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)
    train_indices = contiguous_idx_blocks(indices, args.min_len_block)
    assert train_indices, 'index list empty, try reducing args.min_len_block'
    
    while n < args.batch_size:
        idx_block = train_indices[rnd.randint(len(train_indices))]
        i = rnd.choice(idx_block)
        x = get_x(handle, i)

        if reject(args, model, x, ground_truth_x_fov, ground_truth_x_fov, i, high_pass_filter):
            continue
        else:
            target = get_target(args, int(ground_truth_x_fov[i]), int(ground_truth_y_fov[i]))
            input_frames[n, 0] = x
            targets[n] = target
            n += 1
            
    file_close(handle)

    return input_frames, targets


def load_dataset_prior(args: Mapping[str, Any],
                 path_dict: Mapping[str, Any]) -> Tuple[TensorType, TensorType]:

    input_frames = []
    targets = []
    
    for idx in range(len(path_dict)):

        path_to_file, handle, indices, ground_truth_x_fov, ground_truth_y_fov = get_ground_truth(path_dict, idx)
        path_to_result = get_path_to_result(path_dict, idx)
        result_list = pickle.load(open(f'{path_to_result}', 'rb'))
        idx_list_baseline, x_difference_baseline, y_difference_baseline, candidate_x_coordinate_list, candidate_y_coordinate_list, peak_list = result_list
        train_indices = find_baseline_false(idx_list_baseline, x_difference_baseline, y_difference_baseline, args.pixel_cutoff)
        assert train_indices, 'no hard frames found'

        for i in train_indices:
            x = get_x(handle, i)
            target = get_target(args, int(ground_truth_x_fov[i]), int(ground_truth_y_fov[i]))
            input_frames.append(x)
            targets.append(target)

        file_close(handle)
        
    input_frames = np.stack(input_frames)
    input_frames = input_frames.reshape(len(input_frames), 1, args.h, args.h)
    targets = np.stack(targets)
        
    return input_frames, targets
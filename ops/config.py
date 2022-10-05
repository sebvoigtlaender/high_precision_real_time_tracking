from typing import Any, Mapping
import numpy as np


def get_config(args: Mapping[str, Any]) -> Mapping[str, Any]:

    args.train_dataset_id = 'train'
    args.cross_val_dataset_id = 'eval'
    args.eval_dataset_id = 'eval'
    args.search_hp_dataset_id = 'eval'
    args.test_dataset_id = 'test'
    args.track_dataset_id = 'test'

    args.state_dict_path = f'results/{args.core_type}'
    args.train_result_path = f'results/train_loss'
    args.cross_val_dict_path = f'data/cross_val_'
    args.test_result_path = f'results/test_result'
    args.search_result_path = f'results/search_hp'
    args.hp = 'hp'
    
    args.train_type = 'basic'
    args.eval_type = 'model-based'
    args.search_hp_type = 'model-free'
    args.test_type = 'test'

    args.correlation_scale = np.logspace(0, -5, 20)
    args.hard_boundary = np.linspace(50, 200, 16)

    return args

def get_path_dict(key: str) -> Mapping[str, Any]:
    path_dict = dict()
    if key == 'train':
        path_dict[0] = 'data/train_dataset.pkl'
    elif key == 'eval':
        path_dict[0] = 'data/eval_dataset.pkl'
    elif key == 'search-hp':
        path_dict[0] = 'data/eval_dataset.pkl'
    elif key == 'test':
        path_dict[0] = 'data/test_dataset.pkl'
    return path_dict
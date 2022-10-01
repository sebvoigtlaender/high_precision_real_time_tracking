from typing import Any, Mapping
import numpy as np


def get_config(args: Mapping[str, Any]) -> Mapping[str, Any]:

    args.train_dataset_id = 'train'
    args.cross_val_dataset_id = 'train'
    args.eval_dataset_id = 'eval'
    args.search_hp_dataset_id = 'eval'
    args.test_dataset_id = 'test'
    args.track_dataset_id = 'test'

    args.state_dict_path = f'results/{args.core_type}'
    args.train_result_path = f'results/train_loss'
    args.cross_val_dict_path = f'data/cross_val_dict'
    args.cross_val_result_path = f'results/cross_val'
    args.test_result_path = f'results/test_result'
    args.search_result_path = f'results/search_hp'
    args.hp = 'results/hp'
    
    args.train_type = 'basic'
    args.eval_type = 'model-based'
    args.search_hp_type = 'model-free'
    args.test_type = 'track'

    args.correlation_scale = np.logspace(0, -5, 20)
    args.hard_boundary = np.linspace(50, 200, 16)

    return args


def get_path_dict(key: str) -> Mapping[str, Any]:
    return_dict = {}
    # if key == 'train':
    #     server_id = [9, 9, 7, 7, 4, 4, 4, 7, 6, 4] + [9]*12
    #     user = ['vikash', 'vikash', 'charlie', 'charlie', 'jen', 'jen', 'jen', 'charlie', 'charlie', 'lilian'] + ['lilian']*12
    #     date_list = ['20220411_145041', '20220411_171726', '20220511_083633', '20220511_103737', '20220407_152537', '20220407_123057', '20220407_141023', '20220512_090435', '20220414_082946', '20210427_093446'] + ['20210622_095759', '20211209_121532', '20210622_121642', '20211201_151012', '20211208_142407', '20211201_124415', '20210610_103157', '20210701_113413', '20211209_094916', '20211207_151916', '20211207_104052', '20211202_145140']
    if key == 'train':
        server_id = [9]*12
        user = ['lilian']*12
        date_list = ['20210622_095759', '20211209_121532', '20210622_121642', '20211201_151012', '20211208_142407', '20211201_124415', '20210610_103157', '20210701_113413', '20211209_094916', '20211207_151916', '20211207_104052', '20211202_145140']
    elif key == 'opt_h':
        date_list = ['20210622_095759', '20211209_121532', '20210622_121642', '20211201_151012']
        server_id = [9]*len(date_list)
        user = ['lilian']*len(date_list)
    elif key == 'test':
        date_list = ['20211130_150801', '20211202_122605']
        # date_list = ['20211130_172651']
        server_id = [9]*len(date_list)
        user = ['lilian']*len(date_list)
    elif key == 'eval':
        date_list = ['20210610_103157', '20210701_113413', '20211202_145140', '20211130_172651']
        server_id = [9]*len(date_list)
        user = ['lilian']*len(date_list)
        
    for i in range(len(date_list)):
        return_dict[i] = {'server_id': server_id[i], 'user': user[i], 'date': date_list[i]}
    
    return return_dict
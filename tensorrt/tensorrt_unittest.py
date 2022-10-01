import time

import numpy as np
import numpy.random as rnd
import scipy.ndimage
from tqdm import tqdm

import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tensorrt_model import Model

gaussian_blur = torchvision.transforms.GaussianBlur(9, 4.0)


def unittest(args, model): 

    with pt.no_grad():
        d_time_list = []
        for t in tqdm(range(2000)):
            x = pt.randn(1, 1, 512, 512).to(args.device)
            t_0 = time.time()
            out = model.infer(x)
            pt.cuda.synchronize()
            t_1 = time.time()
            if t > 10:
                d_time_list.append(t_1-t_0)
            
    return np.mean(d_time_list)


def main():

    import argparse
    import torch
    import torch_tensorrt
    from tensorrt_resnet import resnet18


    parser = argparse.ArgumentParser(description='tracking')
    parser.add_argument('--batch-size', type=int, default=2, help='training batch size (default: 2)')
    parser.add_argument('--core-type', type=str, default='resnet_18', help='model core type (default: resnet-18)')
    parser.add_argument('--device', default='cpu', help='device (default: cpu)')
    parser.add_argument('--h', type=int, default=512, help='input height (default: 512)')
    parser.add_argument('--n-episodes', type=int, default=int(10000), help='number of training episodes (default: 10000)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    args = parser.parse_args(args = [])
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.precision = 'fp16'
    args.device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    
    model = Model(args).to(args.device)
    d_time_list = unittest(args, model)
    print('average inference time: ', d_time_list)

    
if __name__  == '__main__':
    main()
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch as pt
import torch.nn as nn
import torch_tensorrt

from tensorrt_resnet import resnet18
from tensorrt_fpn import FeaturePyramidNetwork

# from utils import gaussian_blur

class Core():
    
    def __init__(self, args: Mapping[str, Any]) -> None:
        super().__init__()
        self.args = args
        self.input = pt.nn.Conv2d(1, 3, kernel_size=3, stride=2)
        self.resnet_core = resnet18()

class Model(nn.Module):

    def __init__(self, args: Mapping[str, Any]) -> None:
        super().__init__()
        self.args = args
        self.input = pt.nn.Conv2d(1, 3, kernel_size=3, stride=2).eval()
        self.resnet_core = resnet18().eval()
        self.fpn = FeaturePyramidNetwork([1, 3, 64, 128, 256, 512], 1).to(args.device).eval()

        if args.precision == 'fp32':
            self.trt_core = torch_tensorrt.compile(self.resnet_core, inputs = [torch_tensorrt.Input((1, 3, 255, 255), dtype=pt.float32)], enabled_precisions = pt.float32)

        elif args.precision == 'fp16':
            self.trt_core = torch_tensorrt.compile(self.resnet_core, inputs = [torch_tensorrt.Input((1, 3, 255, 255), dtype=pt.half)], enabled_precisions = pt.half)
            self.input = self.input.half()
            self.fpn = self.fpn.half()
        

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        input_out = self.input(x)
        out_1, out_2, out_3, out_4 = self.trt_core(input_out)
        out = self.fpn(x, input_out, out_1, out_2, out_3, out_4)
        return out


    def infer(self, x: List) -> pt.Tensor:
        with pt.no_grad():
            if self.args.precision == 'fp16':
                x = x.half()
            out = self(x)
            out = pt.sigmoid(out)
            return out
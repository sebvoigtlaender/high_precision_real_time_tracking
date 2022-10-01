from typing import Tuple, List, Dict, Optional

import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops.feature_pyramid_network import ExtraFPNBlock


class FeaturePyramidNetwork(nn.Module):

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[int] = None,
    ):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)


    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out


    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out
        


    def forward(self, x: Tensor, out_input: Tensor, out_1: Tensor, out_2: Tensor, out_3: Tensor, out_4: Tensor) -> Tensor:

        last_inner_4 = self.get_result_from_inner_blocks(out_4, -1)

        idx = 4
        inner_lateral_4 = self.get_result_from_inner_blocks(out_3, idx)
        feat_shape_4 = inner_lateral_4.shape[-2:]
        inner_top_down_4 = F.interpolate(last_inner_4, size=feat_shape_4, mode="nearest")
        last_inner_3 = inner_lateral_4 + inner_top_down_4
        out_4 = self.get_result_from_layer_blocks(last_inner_3, idx)

        idx = 3
        inner_lateral_3 = self.get_result_from_inner_blocks(out_2, idx)
        feat_shape_3 = inner_lateral_3.shape[-2:]
        inner_top_down_3 = F.interpolate(last_inner_3, size=feat_shape_3, mode="nearest")
        last_inner_2 = inner_lateral_3 + inner_top_down_3
        out_3 = self.get_result_from_layer_blocks(last_inner_2, idx)

        idx = 2
        inner_lateral_2 = self.get_result_from_inner_blocks(out_1, idx)
        feat_shape_2 = inner_lateral_2.shape[-2:]
        inner_top_down_2 = F.interpolate(last_inner_2, size=feat_shape_2, mode="nearest")
        last_inner_1 = inner_lateral_2 + inner_top_down_2
        out_2 = self.get_result_from_layer_blocks(last_inner_1, idx)

        idx = 1
        inner_lateral_1 = self.get_result_from_inner_blocks(out_input, idx)
        feat_shape_1 = inner_lateral_1.shape[-2:]
        inner_top_down_1 = F.interpolate(last_inner_1, size=feat_shape_1, mode="nearest")
        last_inner_0 = inner_lateral_1 + inner_top_down_1
        out_1 = self.get_result_from_layer_blocks(last_inner_0, idx)

        idx = 0
        inner_lateral_0 = self.get_result_from_inner_blocks(x, idx)
        feat_shape_0 = inner_lateral_0.shape[-2:]
        inner_top_down_0 = F.interpolate(last_inner_0, size=feat_shape_0, mode="nearest")
        last_inner_final = inner_lateral_0 + inner_top_down_0
        out_0 = self.get_result_from_layer_blocks(last_inner_final, idx)

        return out_0
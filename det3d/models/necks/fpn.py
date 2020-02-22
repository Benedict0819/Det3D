import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core import auto_fp16
from det3d.torchie.cnn import xavier_init
from det3d.models.utils import Empty, GroupNorm, Sequential

from ..registry import NECKS
from ..utils import ConvModule, build_norm_layer
from ..backbones import ResNet

@NECKS.register_module
class FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        strides,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        extra_convs_on_inputs=True,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        activation=None,
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.strides = strides
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False,
                )
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    # @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=self.strides[i], mode="nearest"
            )

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@NECKS.register_module
class ResNet_FPN(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        num_input_features,
        fpn_num_filters,
        norm_cfg=None,
        name="resnet_fpn",
        logger=None,
        include_stem_layer=False,
        **kwargs
    ):
        super(ResNet_FPN, self).__init__()

        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features
        self._fpn_num_filters = fpn_num_filters

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        self.resnet = ResNet(
            block_type="basic",
            include_stem_layer=include_stem_layer,
            num_input_features=self._num_input_features,
            num_stages=len(layer_nums),
            block_nums=layer_nums,
            norm_cfg=self._norm_cfg,
            strides=ds_layer_strides,
            n_planes=ds_num_filters,
            dilations=[1]*len(layer_nums),
            out_indices=list(range(len(layer_nums))),
            stage_with_gcb=[False]*len(layer_nums),
            stage_with_dcn=[False]*len(layer_nums),
            stage_with_gen_attention=[()]*len(layer_nums),
            )
        logger.info("Finished initializing resnet.")

        self.fpn = FPN(       
            in_channels=ds_num_filters,
            out_channels=fpn_num_filters,
            num_outs=len(ds_num_filters),
            strides=ds_layer_strides)

        logger.info("Finished initializing fpn.")

    def forward(self, x):
        feats = self.resnet(x)
        feats = self.fpn(feats)
        return feats

@NECKS.register_module
class ResNet_Panoptic_FPN(ResNet_FPN):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        fpn_num_filters,
        norm_cfg=None,
        aggregation_method="concat",
        include_stem_layer=False,
        name="resnet_panoptic_fpn",
        logger=None,
        **kwargs
    ):

        super(ResNet_Panoptic_FPN, self).__init__(
            layer_nums=layer_nums,
            ds_layer_strides=ds_layer_strides,
            ds_num_filters=ds_num_filters,
            num_input_features=num_input_features,
            fpn_num_filters=fpn_num_filters,
            include_stem_layer=include_stem_layer,
            norm_cfg=norm_cfg,
            logger=logger,
            **kwargs
        )

        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._aggregation_method = aggregation_method

        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            num_out_filters = self._fpn_num_filters
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        logger.info("Finished initializing panoptic-fpn.")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def forward(self, x):

        x = self.resnet(x)
        x = self.fpn(x)
        ups=[]
        for i in range(len(self.deblocks)):
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x[i]))
        if len(ups) > 0:
            if self._aggregation_method == "add":
                x = torch.stack(ups, dim=0).sum(dim=0)
            else:
                x = torch.cat(ups, dim=1)

        return x

    def _test(self):
        x = torch.randn((1, 64, 512, 512))
        return self.forward(x)
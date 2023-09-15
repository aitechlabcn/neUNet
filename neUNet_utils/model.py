import math

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp, convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from typing import Union, List, Tuple, Type
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import numpy as np
from SRUNet.MultiscaleEncoder import MultiscaleEncoder


class PixelShuffle3d(nn.Module):
    """
    This class is a 3d version of pixelshuffle.
    """

    def __init__(self, upscale_factor, out_channels):
        """
        :param scale: upsample scale
        """
        super().__init__()
        self.scale = upscale_factor
        self.out_channels = out_channels

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        self.scale = np.array(self.scale)
        n = np.sum(self.scale == 2)
        assert channels % (2 ** n) == 0
        nOut = self.out_channels

        out_depth = in_depth * self.scale[0]
        out_height = in_height * self.scale[1]
        out_width = in_width * self.scale[2]

        input_view = x.contiguous().view(batch_size, nOut, self.scale[0], self.scale[1], self.scale[2], in_depth,
                                         in_height,
                                         in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class ConvWithPixelShuffle(nn.Module):
    def __init__(self, conv_op, in_channels, out_channels, scales):
        super().__init__()
        self.dims = convert_conv_op_to_dim(conv_op)
        n = np.prod(scales)
        self.conv1 = conv_op(in_channels, n/2 * in_channels, 5, 1, 2)
        self.tanh = nn.Tanh()
        if self.dims == 3:
            self.conv2 = conv_op(n/2 * in_channels, out_channels * n, 3, 1, 1)
        elif self.dims == 2:
            self.conv2 = conv_op(2 * in_channels, out_channels * n, 3, 1, 1)
        self.pixelshuffle = match_pixel_shuffle(dims=self.dims, scale=scales, out_channels=out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.tanh(y)
        y = self.conv2(y)
        y = self.tanh(y)
        y = self.pixelshuffle(y)
        return y


def match_pixel_shuffle(dims, scale, out_channels):
    if dims == 2:
        return nn.PixelShuffle(upscale_factor=scale)
    if dims == 3:
        return PixelShuffle3d(upscale_factor=scale, out_channels=out_channels)


class SRUnetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        stages = []
        convswithpixelshuffle = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            convswithpixelshuffle.append(ConvWithPixelShuffle(
                conv_op=encoder.conv_op, in_channels=input_features_below, out_channels=input_features_skip,
                scales=stride_for_transpconv
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.convswithpixelshuffle = nn.ModuleList(convswithpixelshuffle)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.convswithpixelshuffle[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if convert_conv_op_to_dim(self.encoder.conv_op) == 3:
                output += np.prod([self.encoder.output_channels[-(s + 2)] * 12 / 8, *skip_sizes[-(s + 1)]],
                                  dtype=np.int64)
            else:
                output += np.prod([self.encoder.output_channels[-(s + 2)] * 8 / 8, *skip_sizes[-(s + 1)]],
                                  dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class SRUNet(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 wavalet_scales: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first
        )
        self.encoder = MultiscaleEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                         n_conv_per_stage, wavalet_scales, conv_bias, norm_op, norm_op_kwargs,
                                         dropout_op,
                                         dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                         nonlin_first=nonlin_first)
        self.decoder = SRUnetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder,
                                     deep_supervision, nonlin_first=nonlin_first)
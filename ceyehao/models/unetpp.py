"""
unet++ implementation adapted from https://github.com/MrGiovanni/UNetPlusPlus.git
"""

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from torch import nn
import torch
import numpy as np


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=True)
    return data


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # if we have 5 pooling then our patch size must be divisible by 2**5
        self.input_shape_must_be_divisible_by = (
            None  # for example in a 2d network that does 5 pool in x and 6 pool
        )
        # in y this would be (32, 64)

        # we need to know this because we need to know if we are a 2d or a 3d netowrk
        self.conv_op = None  # nn.Conv2d or nn.Conv3d

        # this tells us how many channely we have in the output. Important for preallocation in inference
        self.num_classes = None  # number of channels in the output

        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
    ):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {"p": 0.5, "inplace": True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}
        if conv_kwargs is None:
            conv_kwargs = {
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "bias": True,
            }

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if (
            self.dropout_op is not None
            and self.dropout_op_kwargs["p"] is not None
            and self.dropout_op_kwargs["p"] > 0
        ):
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):  # specify first stride to downsample.
    def __init__(
        self,
        input_feature_channels,
        output_feature_channels,
        num_convs,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        first_stride=None,
        basic_block=ConvDropoutNormNonlin,
    ):
        """
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        """
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {"p": 0, "inplace": True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}
        if conv_kwargs is None:
            conv_kwargs = {
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "bias": True,
            }

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv["stride"] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *(
                [
                    basic_block(
                        input_feature_channels,
                        output_feature_channels,
                        self.conv_op,
                        self.conv_kwargs_first_conv,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    )
                ]
                + [
                    basic_block(
                        output_feature_channels,
                        output_feature_channels,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    )
                    for _ in range(num_convs - 1)
                ]
            )
        )

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if (
        isinstance(module, nn.Conv2d)
        or isinstance(module, nn.Conv3d)
        or isinstance(module, nn.Dropout3d)
        or isinstance(module, nn.Dropout2d)
        or isinstance(module, nn.Dropout)
        or isinstance(module, nn.InstanceNorm3d)
        or isinstance(module, nn.InstanceNorm2d)
        or isinstance(module, nn.InstanceNorm1d)
        or isinstance(module, nn.BatchNorm2d)
        or isinstance(module, nn.BatchNorm3d)
        or isinstance(module, nn.BatchNorm1d)
    ):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=False
    ):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class Generic_UNetPlusPlus(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 512

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000 * 2  # 505789440

    def __init__(
        self,
        input_channels,
        base_num_features,
        num_classes,
        num_pool,
        profile_size,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        dropout_in_localization=False,  # turned off deep supervision
        weightInitializer=InitWeights_He(1e-2),
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        convolutional_pooling=False,
        convolutional_upsampling=True,
        max_num_features=None,
        basic_block=ConvDropoutNormNonlin,
    ):

        super(Generic_UNetPlusPlus, self).__init__()
        self.output_size = profile_size

        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        if nonlin_kwargs is None:
            nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {"p": 0, "inplace": True}  # turn off dropout
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}

        self.conv_kwargs = {"stride": 1, "dilation": 1, "bias": True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes

        if conv_op == nn.Conv2d:
            self.upsample_mode = "bilinear"
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        else:
            raise ValueError(
                "unknown convolution dimensionality, conv op: %s" % str(conv_op)
            )

        self.input_shape_must_be_divisible_by = np.prod(
            pool_op_kernel_sizes, 0, dtype=np.int64
        )
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.loc0 = []
        self.loc1 = []
        self.loc2 = []
        self.loc3 = []
        self.td = []
        self.up0 = []
        self.up1 = []
        self.up2 = []
        self.up3 = []
        self.seg_outputs = []

        output_features = base_num_features  # channels of x00. channels of x_n0 is multiplied by the factor feat_map_mul_on_downscale
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride when conv pool is used
            if (
                d != 0 and self.convolutional_pooling
            ):  # by defult conv pool is false d=0 is the X00, no downscaling.
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None  # specify the stride of the first conv layer in the stacked conv layers. none make stride = 1

            self.conv_kwargs["kernel_size"] = self.conv_kernel_sizes[d]
            self.conv_kwargs["padding"] = self.conv_pad_sizes[d]
            # add downsample-convolutions
            self.conv_blocks_context.append(
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    first_stride,
                    basic_block=basic_block,
                )
            )
            if not self.convolutional_pooling:
                self.td.append(
                    pool_op(pool_op_kernel_sizes[d])
                )  # the first stack conv dont need td.

            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None  # the last layer

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs["kernel_size"] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs["padding"] = self.conv_pad_sizes[num_pool]  # auto-setup
        self.conv_blocks_context.append(
            nn.Sequential(  # these two forms a single stacked conv layer first part do the stride, and second part do the ajust feature numbers, but why break it into 2 parts?
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage - 1,
                    self.conv_op,
                    self.conv_kwargs, 
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    first_stride,
                    basic_block=basic_block,
                ),
                StackedConvLayers(
                    output_features,
                    final_num_features,
                    1,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    basic_block=basic_block,
                ),
            )
        )

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs["p"]
            self.dropout_op_kwargs["p"] = 0.0

        # now lets build the localization pathway
        encoder_features = (
            final_num_features  # number of input features of the kernels.
        )
        self.loc0, self.up0, encoder_features = self.create_nest(
            0, num_pool, final_num_features, num_conv_per_stage, basic_block, transpconv
        )
        self.loc1, self.up1, encoder_features1 = self.create_nest(
            1, num_pool, encoder_features, num_conv_per_stage, basic_block, transpconv
        )
        self.loc2, self.up2, encoder_features2 = self.create_nest(
            2, num_pool, encoder_features1, num_conv_per_stage, basic_block, transpconv
        )
        self.loc3, self.up3, encoder_features3 = self.create_nest(
            3, num_pool, encoder_features2, num_conv_per_stage, basic_block, transpconv
        )
        if not dropout_in_localization:
            self.dropout_op_kwargs["p"] = old_dropout_p

        # register all modules properly
        self.loc0 = nn.ModuleList(self.loc0)
        self.loc1 = nn.ModuleList(self.loc1)
        self.loc2 = nn.ModuleList(self.loc2)
        self.loc3 = nn.ModuleList(self.loc3)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.up0 = nn.ModuleList(self.up0)
        self.up1 = nn.ModuleList(self.up1)
        self.up2 = nn.ModuleList(self.up2)
        self.up3 = nn.ModuleList(self.up3)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

        if self.output_size == [200, 200]:
            self.align = nn.Conv2d(
                base_num_features, 2, kernel_size=7, stride=1, padding=3
            )  # out 200*200. not in use, but keep it here to load param dict and exclude from the forward compute.
        elif self.output_size == [100, 100]:
            self.align = nn.Conv2d(
                base_num_features, 2, kernel_size=5, stride=2, padding=2
            )  # out 100*100
        elif self.output_size == [50, 50]:
            self.align = torch.nn.Sequential(
                nn.Conv2d(
                    base_num_features, 2, kernel_size=5, stride=2, padding=2
                ),  # out 100*100
                nn.Conv2d(
                    base_num_features, 2, kernel_size=3, stride=2, padding=1
                ),  # out 50*50
            )
        else:
            raise ValueError(
                f"output resolution is {self.output_size}, and not supported!"
            )

    def forward(self, x):
        x0_0 = self.conv_blocks_context[0](x)
        x1_0 = self.conv_blocks_context[1](self.td[0](x0_0))
        x0_1 = self.loc3[0](torch.cat([x0_0, self.up3[0](x1_0)], 1))

        x2_0 = self.conv_blocks_context[2](self.td[1](x1_0))
        x1_1 = self.loc2[0](torch.cat([x1_0, self.up2[0](x2_0)], 1))
        x0_2 = self.loc2[1](torch.cat([x0_0, x0_1, self.up2[1](x1_1)], 1))

        x3_0 = self.conv_blocks_context[3](self.td[2](x2_0))
        x2_1 = self.loc1[0](torch.cat([x2_0, self.up1[0](x3_0)], 1))
        x1_2 = self.loc1[1](torch.cat([x1_0, x1_1, self.up1[1](x2_1)], 1))
        x0_3 = self.loc1[2](torch.cat([x0_0, x0_1, x0_2, self.up1[2](x1_2)], 1))

        x4_0 = self.conv_blocks_context[4](self.td[3](x3_0))
        x4_0_up = self.match_size(self.up0[0](x4_0), x3_0)
        x3_1 = self.loc0[0](torch.cat([x3_0, x4_0_up], 1))
        x2_2 = self.loc0[1](torch.cat([x2_0, x2_1, self.up0[1](x3_1)], 1))
        x1_3 = self.loc0[2](torch.cat([x1_0, x1_1, x1_2, self.up0[2](x2_2)], 1))
        x0_4 = self.loc0[3](torch.cat([x0_0, x0_1, x0_2, x0_3, self.up0[3](x1_3)], 1))
        out = self.align(x0_4)
        return out

    # now lets build the localization pathway BACK_UP
    def create_nest(
        self,
        z,
        num_pool,
        final_num_features,
        num_conv_per_stage,
        basic_block,
        transpconv,
    ):  # num pool =5 following the original naming of locZ and upZ
        """

        Args:
            z (_type_):
            num_pool (_type_):
            final_num_features (_type_): feature number to be upsampled.
            num_conv_per_stage (_type_):
            basic_block (_type_):
            transpconv (_type_):
        """
        conv_blocks_localization = []
        tu = []
        i = 0
        for u in range(z, num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)
            ].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2 # loc does not change numner of features
            n_features_after_tu_and_concat = nfeatures_from_skip * (
                2 + u - z
            )  # u==z is loc_z0, only 1 feature map from skip connection and always 1 feature map from upsampled feature map.
            if i == 0:  # when u==z
                unet_final_features = nfeatures_from_skip
                i += 1
            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip  # upsample result has the same number of features as the feat map from skip connection .

            if not self.convolutional_upsampling:
                tu.append(
                    Upsample(
                        scale_factor=self.pool_op_kernel_sizes[-(u + 1)],
                        mode=self.upsample_mode,
                    )
                )
            else:
                tu.append(
                    transpconv(
                        nfeatures_from_down,
                        nfeatures_from_skip,
                        self.pool_op_kernel_sizes[-(u + 1)],
                        self.pool_op_kernel_sizes[-(u + 1)],
                        bias=False,
                    )
                )

            self.conv_kwargs["kernel_size"] = self.conv_kernel_sizes[-(u + 1)]
            self.conv_kwargs["padding"] = self.conv_pad_sizes[-(u + 1)]
            conv_blocks_localization.append(
                nn.Sequential(
                    StackedConvLayers(
                        n_features_after_tu_and_concat,
                        nfeatures_from_skip,
                        num_conv_per_stage - 1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        basic_block=basic_block,
                    ),
                    StackedConvLayers(
                        nfeatures_from_skip,
                        final_num_features,
                        1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        basic_block=basic_block,
                    ),
                )
            )
        return conv_blocks_localization, tu, unet_final_features

    def match_size(self, upsampled_feat, feat_from_skip):
        if (
            upsampled_feat.shape[2] != feat_from_skip.shape[2]
            or upsampled_feat.shape[3] != feat_from_skip.shape[3]
        ):
            diff2 = feat_from_skip.shape[2] - upsampled_feat.shape[2]
            diff3 = feat_from_skip.shape[3] - upsampled_feat.shape[3]
            upsampled_feat = nn.functional.pad(
                upsampled_feat,
                (diff3 // 2, diff3 - diff3 // 2, diff2 // 2, diff2 - diff2 // 2),
            )

        return upsampled_feat

    @staticmethod
    def compute_approx_vram_consumption(
        patch_size,
        num_pool_per_axis,
        base_num_features,
        max_num_features,
        num_modalities,
        num_classes,
        pool_op_kernel_sizes,
        deep_supervision=False,
        conv_per_stage=2,
    ):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64(
            (conv_per_stage * 2 + 1)
            * np.prod(map_size, dtype=np.int64)
            * base_num_features
            + num_modalities * np.prod(map_size, dtype=np.int64)
            + num_classes * np.prod(map_size, dtype=np.int64)
        )

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (
                (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage
            )  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp

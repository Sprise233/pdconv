from models.utils.BasicConv import BasicConv3D, BasicConv
from models.utils.PDConv import PDConv_A, PDConv_B, PDConv_C, PDConv_D
import torch.nn as nn

class MedNeXt_meta(BasicConv):
    def __init__(self, conv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = conv

        self.conv1 = BasicConv3D(input_channels=self.input_channels, output_channels=self.output_channels * 2,
                                 kernel_size=1, initial_stride=1, nonlin=nn.GELU, nonlin_kwargs={})
        self.conv2 = BasicConv3D(input_channels=self.output_channels * 2, output_channels=self.output_channels,
                                 kernel_size=1, initial_stride=1)

        self.is_skip_connection = all(
            s == 1 for s in self.initial_stride) and self.input_channels == self.output_channels

    def forward(self, x):
        if self.is_skip_connection:
            return self.conv2(self.conv1(self.conv(x))) + x
        else:
            return self.conv2(self.conv1(self.conv(x)))

    def compute_conv_feature_map_size(self, input_size):
        output = 0
        output += self.conv1.compute_conv_feature_map_size(input_size)
        output += self.conv2.compute_conv_feature_map_size(input_size)
        output += self.conv.compute_conv_feature_map_size(input_size)
        return output


class MedNeXt_A(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pd_conv = PDConv_A(input_channels=self.input_channels, output_channels=self.input_channels,
                                is_dw_conv=True, norm_op=nn.InstanceNorm3d,
                                norm_op_kwargs={"eps": 1e-05, "affine": True},
                                initial_stride=self.initial_stride)

        self.mednext = MedNeXt_meta(self.pd_conv, *args, **kwargs)

    def forward(self, x):
        return self.mednext(x)

    def compute_conv_feature_map_size(self, input_size):
        output = self.mednext.compute_conv_feature_map_size(input_size)
        return output

class MedNeXt_B(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pd_conv = PDConv_B(input_channels=self.input_channels, output_channels=self.input_channels,
                                is_dw_conv=True, norm_op=nn.InstanceNorm3d,
                                norm_op_kwargs={"eps": 1e-05, "affine": True},
                                initial_stride=self.initial_stride)

        self.mednext = MedNeXt_meta(self.pd_conv, *args, **kwargs)

    def forward(self, x):
        return self.mednext(x)

    def compute_conv_feature_map_size(self, input_size):
        output = self.mednext.compute_conv_feature_map_size(input_size)
        return output

class MedNeXt_C(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pd_conv = PDConv_C(input_channels=self.input_channels, output_channels=self.input_channels,
                                is_dw_conv=True, norm_op=nn.InstanceNorm3d,
                                norm_op_kwargs={"eps": 1e-05, "affine": True},
                                initial_stride=self.initial_stride)

        self.mednext = MedNeXt_meta(self.pd_conv, *args, **kwargs)

    def forward(self, x):
        return self.mednext(x)

    def compute_conv_feature_map_size(self, input_size):
        output = self.mednext.compute_conv_feature_map_size(input_size)
        return output

class MedNeXt_D(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pd_conv = PDConv_D(input_channels=self.input_channels, output_channels=self.input_channels,
                                is_dw_conv=True, norm_op=nn.InstanceNorm3d,
                                norm_op_kwargs={"eps": 1e-05, "affine": True},
                                initial_stride=self.initial_stride)

        self.mednext = MedNeXt_meta(self.pd_conv, *args, **kwargs)

    def forward(self, x):
        return self.mednext(x)

    def compute_conv_feature_map_size(self, input_size):
        output = self.mednext.compute_conv_feature_map_size(input_size)
        return output

class MedNeXt_3D(BasicConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pd_conv = BasicConv3D(input_channels=self.input_channels, output_channels=self.input_channels,
                                is_dw_conv=True, norm_op=nn.InstanceNorm3d,
                                norm_op_kwargs={"eps": 1e-05, "affine": True},
                                initial_stride=self.initial_stride)

        self.mednext = MedNeXt_meta(self.pd_conv, *args, **kwargs)

    def forward(self, x):
        return self.mednext(x)

    def compute_conv_feature_map_size(self, input_size):
        output = self.mednext.compute_conv_feature_map_size(input_size)
        return output


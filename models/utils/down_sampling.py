import torch
from dynamic_network_architectures.building_blocks.regularization import DropPath, SqueezeExcite
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from models.utils.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from models.utils.simple_conv_blocks import ConvDropoutNormReLU
import torch.nn.functional as F


class HaarDownSampling3D(nn.Module):
    """
    使用 3D Haar 核实现高效的 3D 小波下采样。
    参数:
        input_channels: 输入张量的通道数
        stride: 下采样步长 (stride_l, stride_h, stride_w)
    """

    def __init__(self, input_channels, stride, learnable=False, device=None, dtype=None, is_dw_conv=True):
        super(HaarDownSampling3D, self).__init__()
        self.stride = stride
        self.learnable = learnable
        self.is_dw_conv = is_dw_conv

        # stride = [1 if i == 1 else i * 3 + 1 for i in stride]
        # 计算各个维度的填充
        self.padding_l = (stride[0] - 1) // 2
        self.padding_h = (stride[1] - 1) // 2
        self.padding_w = (stride[2] - 1) // 2

        if learnable:
            # 获取当前设备的默认数据类型（支持 float16、float32 等）
            factory_kwargs = {'device': device, 'dtype': dtype}

            # 将核变为可学习的参数，数据类型自适应
            self.low_l = nn.Parameter(torch.ones((input_channels, 1 if is_dw_conv else input_channels, stride[0], 1, 1), **factory_kwargs))
            self.low_h = nn.Parameter(torch.ones((input_channels, 1 if is_dw_conv else input_channels, 1, stride[1], 1), **factory_kwargs))
            self.low_w = nn.Parameter(torch.ones((input_channels, 1 if is_dw_conv else input_channels, 1, 1, stride[2]), **factory_kwargs))

            # 将高通滤波器核也变为可学习的参数，数据类型自适应
            self.high_l = torch.zeros((input_channels, 1 if is_dw_conv else input_channels, stride[0], 1, 1), **factory_kwargs) - 1
            self.high_l.data[:, 0, 0, 0, 0] = 1  # 初始化为 Haar 核的高通滤波器
            self.high_l = nn.Parameter(self.high_l)

            self.high_h = torch.zeros((input_channels, 1 if is_dw_conv else input_channels, 1, stride[1], 1), **factory_kwargs) - 1
            self.high_h.data[:, 0, 0, 0, 0] = 1  # 初始化为 Haar 核的高通滤波器
            self.high_h = nn.Parameter(self.high_h)

            self.high_w = torch.zeros((input_channels, 1 if is_dw_conv else input_channels, 1, 1, stride[2]), **factory_kwargs) - 1
            self.high_w.data[:, 0, 0, 0, 0] = 1  # 初始化为 Haar 核的高通滤波器
            self.high_w = nn.Parameter(self.high_w)
        else:
            self.low_l = torch.ones((input_channels, 1 if is_dw_conv else input_channels, stride[0], 1, 1))
            self.low_h = torch.ones((input_channels, 1 if is_dw_conv else input_channels, 1, stride[1], 1))
            self.low_w = torch.ones((input_channels, 1 if is_dw_conv else input_channels, 1, 1, stride[2]))

            self.high_l = torch.zeros((input_channels, 1 if is_dw_conv else input_channels, stride[0], 1, 1)) - 1
            self.high_l[:, 0, 0, 0, 0] = 1

            self.high_h = torch.zeros((input_channels, 1 if is_dw_conv else input_channels, 1, stride[1], 1)) - 1
            self.high_h[:, 0, 0, 0, 0] = 1

            self.high_w = torch.zeros((input_channels, 1 if is_dw_conv else input_channels, 1, 1, stride[2])) - 1
            self.high_w[:, 0, 0, 0, 0] = 1



    def forward(self, x):
        """
        前向传播，进行 3D Haar 小波下采样。

        x: 输入张量，形状为 (B, C, L, H, W)

        返回：下采样后的张量，形状为 (B, C, L//stride_l, H//stride_h, W//stride_w)
        """
        input_tensor = x
        b, c, l, h, w = input_tensor.shape
        stride = self.stride
        if not self.learnable:
            device, dtype = x.device, x.dtype
            self.low_l = self.low_l.to(device=device, dtype=dtype)
            self.low_h = self.low_h.to(device=device, dtype=dtype)
            self.low_w = self.low_w.to(device=device, dtype=dtype)
            self.high_l = self.high_l.to(device=device, dtype=dtype)
            self.high_h = self.high_h.to(device=device, dtype=dtype)
            self.high_w = self.high_w.to(device=device, dtype=dtype)

        def low_transform_l(input_tensor):
            return F.conv3d(input_tensor, self.low_l, stride=(stride[0], 1, 1), padding=(self.padding_l, 0, 0), groups=c if self.is_dw_conv else 1)

        def low_transform_h(input_tensor):
            return F.conv3d(input_tensor, self.low_h, stride=(1, stride[1], 1), padding=(0, self.padding_h, 0), groups=c if self.is_dw_conv else 1)

        def low_transform_w(input_tensor):
            return F.conv3d(input_tensor, self.low_w, stride=(1, 1, stride[2]), padding=(0, 0, self.padding_w), groups=c if self.is_dw_conv else 1)

        def high_transform_l(input_tensor):
            return F.conv3d(input_tensor, self.high_l, stride=(stride[0], 1, 1), padding=(self.padding_l, 0, 0), groups=c if self.is_dw_conv else 1)

        def high_transform_h(input_tensor):
            return F.conv3d(input_tensor, self.high_h, stride=(1, stride[1], 1), padding=(0, self.padding_h, 0), groups=c if self.is_dw_conv else 1)

        def high_transform_w(input_tensor):
            return F.conv3d(input_tensor, self.high_w, stride=(1, 1, stride[2]), padding=(0, 0, self.padding_w), groups=c if self.is_dw_conv else 1)

        haar_list = []
        if stride[0] != 1:
            if len(haar_list) == 0:
                haar_list.append(low_transform_l(input_tensor))
                haar_list.append(high_transform_l(input_tensor))
            else:
                harr_list_temp = []
                for haar in haar_list:
                    harr_list_temp.append(low_transform_l(haar))
                    harr_list_temp.append(high_transform_l(haar))
                haar_list = harr_list_temp

        if stride[1] != 1:
            if len(haar_list) == 0:
                haar_list.append(low_transform_h(input_tensor))
                haar_list.append(high_transform_h(input_tensor))
            else:
                harr_list_temp = []
                for haar in haar_list:
                    harr_list_temp.append(low_transform_h(haar))
                    harr_list_temp.append(high_transform_h(haar))
                haar_list = harr_list_temp

        if stride[2] != 1:
            if len(haar_list) == 0:
                haar_list.append(low_transform_w(input_tensor))
                haar_list.append(high_transform_w(input_tensor))
            else:
                harr_list_temp = []
                for haar in haar_list:
                    harr_list_temp.append(low_transform_w(haar))
                    harr_list_temp.append(high_transform_w(haar))
                haar_list = harr_list_temp

        haar_list = [haar / (stride[0] * stride[1] * stride[2]) for haar in haar_list]

        if len(haar_list) == 0:
            haar_list.append(input_tensor)

        return haar_list


def haar_downsample_3d_efficient(input_tensor, stride=(2, 2, 2)):

    # Haar 核权重，形状为 (1, 1, 2, 2, 2)
    haar_kernel_lll = torch.ones((1, 1, *stride), dtype=input_tensor.dtype,
                                 device=input_tensor.device)

    haar_kernel_lll = haar_kernel_lll / torch.sqrt(
        torch.tensor(stride[0] * stride[1] * stride[2], dtype=input_tensor.dtype, device=input_tensor.device))

    # 使用 3D 卷积进行低频下采样
    b, c, l, h, w = input_tensor.shape
    lll = F.conv3d(
        input_tensor,
        haar_kernel_lll.expand(c, 1, *haar_kernel_lll.shape[2:]),  # 对每个通道使用相同核
        stride=stride,
        padding=0,
        groups=c  # 每个通道独立卷积
    )

    # Haar 核权重，形状为 (1, 1, 2, 2, 2)
    haar_kernel_hhh = torch.zeros((1, 1, *stride), dtype=input_tensor.dtype,
                                  device=input_tensor.device)
    haar_kernel_hhh.fill_(-1)  # 先填充-1
    haar_kernel_hhh[0, 0, 0, 0, 0] = 1  # 将第一个元素设置为1

    haar_kernel_hhh = haar_kernel_hhh / torch.sqrt(
        torch.tensor(stride[0] * stride[1] * stride[2], dtype=input_tensor.dtype, device=input_tensor.device))

    # 使用 3D 卷积进行低频下采样
    b, c, l, h, w = input_tensor.shape
    hhh = F.conv3d(
        input_tensor,
        haar_kernel_hhh.expand(c, 1, *haar_kernel_hhh.shape[2:]),  # 对每个通道使用相同核
        stride=stride,
        padding=0,
        groups=c  # 每个通道独立卷积
    )

    return hhh + lll


class BasicBlockD(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 # todo wideresnet?
                 ):
        """
        This implementation follows ResNet-D:

        He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

        The skip has an avgpool (if needed) followed by 1x1 conv instead of just a strided 1x1 conv

        :param conv_op:
        :param input_channels:
        :param output_channels:
        :param kernel_size: refers only to convs in feature extraction path, not to 1x1x1 conv in skip
        :param stride: only applies to first conv (and skip). Second conv always has stride 1
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op: only the first conv can have dropout. The second never has
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param stochastic_depth_p:
        :param squeeze_excitation:
        :param squeeze_excitation_reduction_ratio:
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, output_channels, output_channels, kernel_size, 1, conv_bias, norm_op,
                                         norm_op_kwargs, None, None, None, None)

        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        # Stochastic Depth
        self.apply_stochastic_depth = False if stochastic_depth_p == 0.0 else True
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1, False, norm_op,
                                        norm_op_kwargs, None, None, None, None
                                        )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin2(out)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any(
                [i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_skip


class BottleneckD(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 bottleneck_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        """
        This implementation follows ResNet-D:

        He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

        The stride sits in the 3x3 conv instead of the 1x1 conv!
        The skip has an avgpool (if needed) followed by 1x1 conv instead of just a strided 1x1 conv

        :param conv_op:
        :param input_channels:
        :param output_channels:
        :param kernel_size: only affects the conv in the middle (typically 3x3). The other convs remain 1x1
        :param stride: only applies to the conv in the middle (and skip). Note that this deviates from the canonical
        ResNet implementation where the stride is applied to the first 1x1 conv. (This implementation follows ResNet-D)
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op: only the second (kernel_size) conv can have dropout. The first and last conv (1x1(x1)) never have it
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param stochastic_depth_p:
        :param squeeze_excitation:
        :param squeeze_excitation_reduction_ratio:
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bottleneck_channels = bottleneck_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, bottleneck_channels, 1, 1, conv_bias,
                                         norm_op, norm_op_kwargs, None, None, nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, bottleneck_channels, bottleneck_channels, kernel_size, stride,
                                         conv_bias,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv3 = ConvDropoutNormReLU(conv_op, bottleneck_channels, output_channels, 1, 1, conv_bias, norm_op,
                                         norm_op_kwargs, None, None, None, None)

        self.nonlin3 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        # Stochastic Depth
        self.apply_stochastic_depth = False if stochastic_depth_p == 0.0 else True
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1, False,
                                        norm_op, norm_op_kwargs, None, None, None, None
                                        )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin3(out)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.bottleneck_channels, *input_size], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.bottleneck_channels, *size_after_stride], dtype=np.int64)
        # conv3
        output_size_conv3 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any(
                [i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_conv3 + output_size_skip


class PatchMerge2D(nn.Module):
    def __init__(self, downscale_factor: int):
        super(PatchMerge2D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        r = self.downscale_factor
        assert height % r == 0 and width % r == 0, "height and width must be divisible by downscale_factor"
        out = x.view(batch_size, channels, height // r, r, width // r, r)
        out = out.permute(0, 1, 3, 5, 2, 4).contiguous()  # [B, C, r, r, H/r, W/r]
        out = out.view(batch_size, channels * (r ** 2), height // r, width // r)  # [B, C*r^2, H/r, W/r]
        return out


class PatchMerge3D(nn.Module):
    def __init__(self, stride: tuple):
        super(PatchMerge3D, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        r = self.stride
        assert depth % r[0] == 0 and height % r[1] == 0 and width % r[
            2] == 0, "depth, height and width must be divisible by downscale_factor"
        out = x.view(batch_size, channels, depth // r[0], r[0], height // r[1], r[1], width // r[2], r[2])
        out = out.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()  # [B, C, r, r, r, D/r, H/r, W/r]
        out = out.view(batch_size, channels * r[0] * r[1] * r[2], depth // r[0], height // r[1],
                       width // r[2])  # [B, C*r[0] * r[1] * r[2], D/r[0], H/r[1], W/r[2]]
        return out


if __name__ == '__main__':
    input_tensor = torch.rand((1, 1, 8, 8, 8))
    print(input_tensor[0, 0, :, :, :])
    patchMerge3D = PatchMerge3D(stride=(2, 2, 2))
    print(patchMerge3D(input_tensor)[0, :, 0, 0, 0])
    print(patchMerge3D(input_tensor).shape)
    # input_tensor = torch.rand((1, 1, 100, 100))
    # patchMerge3D = PatchMerge2D(downscale_factor=4)
    # print(patchMerge3D(input_tensor).shape)

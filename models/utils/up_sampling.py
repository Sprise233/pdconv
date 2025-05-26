import SimpleITK
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class PixelShuffle2D(nn.Module):
    def __init__(self, upscale_factor: int):
        super(PixelShuffle2D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        r = self.upscale_factor
        assert channels % (r ** 2) == 0, "channels must be divisible by upscale_factor^2"
        new_channels = channels // (r ** 2)
        out = x.view(batch_size, new_channels, r, r, height, width)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()  # [B, C, H, r, W, r]
        out = out.view(batch_size, new_channels, height * r, width * r)  # [B, C, H*r, W*r]
        return out


class PixelShuffle3D(nn.Module):
    def __init__(self, stride: tuple):
        super(PixelShuffle3D, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        r = self.stride
        assert channels % (r[0] * r[1] * r[2]) == 0, "channels must be divisible by upscale_factor^3"
        new_channels = channels // (r[0] * r[1] * r[2])
        out = x.view(batch_size, new_channels, r[0], r[1], r[2], depth, height, width)
        out = out.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()  # [B, C, D, r, H, r, W, r]
        out = out.view(batch_size, new_channels, depth * r[0], height * r[1], width * r[2])  # [B, C, D*r, H*r, W*r]
        return out

class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f'pixel_shuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)'
            )
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(
                f'pixel_shuffle expects its input\'s \'channel\' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}'
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(
            *x.shape[:-4],
            nOut,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)

# 可视化函数：展示每个深度层
def visualize_tensor(tensor, title="Tensor Visualization"):
    batch_size, channels, depth, height, width = tensor.size()
    fig, axes = plt.subplots(channels, depth, figsize=(depth * 4, channels * 4))
    fig.suptitle(title, fontsize=16)

    for c in range(channels):
        for d in range(depth):
            ax = axes[c][d] if channels > 1 else axes[d]
            ax.imshow(tensor[0, c, d].detach().numpy(), cmap='viridis')
            ax.set_title(f"Channel {c}, Depth {d}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def verify_shuffle(input_tensor, output_tensor, stride):
    batch_size, channels, depth, height, width = input_tensor.size()
    r = stride
    reshaped = input_tensor.view(batch_size, channels // (r[0] * r[1] * r[2]), r[0], r[1], r[2], depth, height, width)
    reshaped = reshaped.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    expected_output = reshaped.view(batch_size, channels // (r[0] * r[1] * r[2]), depth * r[0], height * r[1], width * r[2])
    assert torch.allclose(output_tensor, expected_output), "Output tensor does not match expected values."
    print("Values match perfectly!")

if __name__ == '__main__':
    input_tensor = torch.rand((1, 8, 2, 2, 2))
    print(input_tensor.shape)
    print(input_tensor)
    patchMerge3D = PixelShuffle3D(stride=(2, 2, 2))
    output_tensor = patchMerge3D(input_tensor)
    # print(output_tensor)
    print(output_tensor.shape)
    print(output_tensor)
    # verify_shuffle(input_tensor, output_tensor, (2, 2, 2))
    a = torch.pixel_shuffle(input_tensor[:, :, 1, :, :], upscale_factor=2)
    b = PixelShuffle3D((1,2,2))(input_tensor)[:, :, 1,:,:]
    print(torch.all(a == b))

    # 可视化输入张量
    # visualize_tensor(input_tensor, title="Input Tensor")

    # 可视化输出张量
    # visualize_tensor(output_tensor, title="Output Tensor")
    # input_tensor = torch.rand((1, 16, 25, 25))
    # patchMerge3D = PixelShuffle2D(upscale_factor=4)
    # print(patchMerge3D(input_tensor).shape)
import torch

from models.UXNet_3D.network_backbone import UXNET


class UX_Net(UXNET):
    def __init__(self, input_channels, output_channels):
        super().__init__(
            in_chans=input_channels,
            out_chans=output_channels,
        )

if __name__ == '__main__':
    input_channels = 1
    output_channels = 1
    data = torch.randn((1, input_channels, 96, 96, 96))
    model = UX_Net(input_channels, output_channels)
    print(model(data).shape)
import torch

from models.TransBTS.TransBTS.TransBTS_downsample8x_skipconnection import BTS


class TransBTS(BTS):
    def __init__(self, input_channels, output_channels, patch_size=96):
        patch_dim = 8
        _conv_repr = True
        _pe_type = "learned"
        super().__init__(
            patch_size,
            patch_dim,
            input_channels,
            output_channels,
            embedding_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=4096,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            conv_patch_representation=_conv_repr,
            positional_encoding_type=_pe_type,
        )

if __name__ == '__main__':
    input_channels = 1
    output_channels = 10
    data = torch.randn((1,1,96,96,96))
    model = TransBTS(input_channels, output_channels)
    print(model(data).shape)
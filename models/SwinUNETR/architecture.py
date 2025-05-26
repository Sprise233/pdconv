from monai.networks.nets import SwinUNETR as swinUNETR
import torch
import torch.nn as nn

class SwinUNETR(swinUNETR):
    def __init__(self, img_size, input_channels, output_channels):
        super().__init__(
            img_size=img_size,
            in_channels=input_channels,
            out_channels=output_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            use_checkpoint=False,
        )

if __name__ == '__main__':
    img_size = (32, 96, 96)
    data_shape = (1, 1, *img_size)
    data = torch.rand(data_shape)
    model = SwinUNETR(img_size, 1, 1)
    print(model(data).shape)
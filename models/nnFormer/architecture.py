import torch

from models.nnFormer.neural_network import SegmentationNetwork
import torch.nn as nn

from models.nnFormer.nnFormer_seg import Encoder, Decoder, final_patch_expanding


class nnFormer(SegmentationNetwork):

    def __init__(self, crop_size=[96, 96, 96],
                 embedding_dim=192,
                 input_channels=1,
                 output_channels=14,
                 conv_op=nn.Conv3d,
                 depths=[2, 2, 2, 2],
                 num_heads=[6, 12, 24, 48],
                 patch_size=[2, 4, 4],
                 window_size=[4, 4, 8, 4],
                 deep_supervision=False):

        super(nnFormer, self).__init__()

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = output_channels
        self.conv_op = conv_op

        self.upscale_logits_ops = []

        self.upscale_logits_ops.append(lambda x: x)

        embed_dim = embedding_dim
        depths = depths
        num_heads = num_heads
        patch_size = patch_size
        window_size = window_size
        self.model_down = Encoder(pretrain_img_size=crop_size, window_size=window_size, embed_dim=embed_dim,
                                  patch_size=patch_size, depths=depths, num_heads=num_heads, in_chans=input_channels)
        self.decoder = Decoder(pretrain_img_size=crop_size, embed_dim=embed_dim, window_size=window_size[::-1][1:],
                               patch_size=patch_size, num_heads=num_heads[::-1][1:], depths=depths[::-1][1:])

        self.final = []
        if self.do_ds:

            for i in range(len(depths) - 1):
                self.final.append(final_patch_expanding(embed_dim * 2 ** i, output_channels, patch_size=patch_size))

        else:
            self.final.append(final_patch_expanding(embed_dim, output_channels, patch_size=patch_size))

        self.final = nn.ModuleList(self.final)

    def forward(self, x):

        seg_outputs = []
        skips = self.model_down(x)
        neck = skips[-1]

        out = self.decoder(neck, skips)

        if self.do_ds:
            for i in range(len(out)):
                seg_outputs.append(self.final[-(i + 1)](out[i]))

            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs[-1]


if __name__ == '__main__':
    data = torch.randn((1, 1, 96, 96, 96))
    model = nnFormer()
    model.compile()
    print(model(data).shape)
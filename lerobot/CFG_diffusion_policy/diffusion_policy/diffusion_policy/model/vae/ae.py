import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            # 输入: (3, 84, 84)
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),  # 84 -> 42
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 42 -> 21
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 调整层: 21 -> 22
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # 保持21
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 输出层
            nn.Conv2d(128, latent_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 输入: (B, 3, 84, 84)
        x = self.net(x)
        # 如果输出不是22x22，使用插值调整
        if x.shape[-1] != 22:
            x = F.interpolate(x, size=22, mode='bilinear', align_corners=False)
        return x  # 输出: (B, 3, 22, 22)
class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            # 输入: (3, 22, 22)
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 第一次上采样: 22 -> 44
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),  # 22 -> 44
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 第二次上采样: 44 -> 88
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),  # 44 -> 88
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 调整层: 88 -> 84
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 输出层
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 输入: (B, 3, 22, 22)
        x = self.net(x)
        # 如果输出不是84x84，使用插值调整
        if x.shape[-1] != 84:
            x = F.interpolate(x, size=84, mode='bilinear', align_corners=False)
        return x  # 输出: (B, 3, 84, 84)

class BasicVAE(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 latent_channels=4,
                 channels=128,
                 encoder_multipliers=[1, 2, 4, 4],
                 decoder_multipliers=[4, 4, 2, 1],
                 num_res_blocks=2):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            # channels=channels,
            # channel_multipliers=encoder_multipliers,
            # num_res_blocks=num_res_blocks
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            # channels=channels,
            # channel_multipliers=decoder_multipliers,
            # num_res_blocks=num_res_blocks
        )

    def forward(self, x):
        # 编码
        latent = self.encoder(x)

        # 解码
        reconstructed = self.decoder(latent)

        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def loss_function(self, reconstructed, original):
        """VAE损失函数"""
        # 重建损失
        # import pdb;pdb.set_trace()
        error=(reconstructed-original)**2
        loss=error.sum(dim=[1,2,3])
        # loss=0
        # threshold=0.95
        # mask=(original<threshold).float()
        # # import pdb;pdb.set_trace()
        # mask_reverse=1-mask
        # print(reconstructed.shape)
        # print(original.shape)
        # error=(reconstructed-original)**2*mask
        # loss=error.sum()/mask.sum()
        # error1=(reconstructed-original)**2*mask_reverse
        # error1=error1.sum()/mask_reverse.sum()
        # loss=loss+error1
        return loss
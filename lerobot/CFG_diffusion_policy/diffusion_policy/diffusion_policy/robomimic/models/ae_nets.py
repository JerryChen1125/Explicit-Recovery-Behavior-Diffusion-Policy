import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveEncoder(nn.Module):
    """适配480x640输入的编码器"""
    def __init__(self, in_channels=3, latent_channels=8, base_channels=64):
        super().__init__()
        
        # 下采样路径 - 适配480x640
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样块 - 480x640 -> 240x320 -> 120x160 -> 60x80 -> 30x40 -> 15x20
        self.down1 = DownsampleBlock(base_channels, base_channels)      # 480x640 -> 240x320
        self.down2 = DownsampleBlock(base_channels, base_channels*2)    # 240x320 -> 120x160
        self.down3 = DownsampleBlock(base_channels*2, base_channels*4)  # 120x160 -> 60x80
        self.down4 = DownsampleBlock(base_channels*4, base_channels*8)  # 60x80 -> 30x40
        self.down5 = DownsampleBlock(base_channels*8, base_channels*8)  # 30x40 -> 15x20
        
        # 中间层
        self.mid_conv1 = nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1)
        self.mid_attn = AttentionBlock(base_channels*8)
        self.mid_conv2 = nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1)
        
        # 输出卷积
        self.conv_out = nn.Conv2d(base_channels*8, latent_channels*2, 3, padding=1)
        
    def forward(self, x):
        # 输入: [B, 3, 480, 640]
        x = F.silu(self.conv_in(x))  # [B, 64, 480, 640]
        
        # 下采样
        x = self.down1(x)  # [B, 64, 240, 320]
        x = self.down2(x)  # [B, 128, 120, 160]
        x = self.down3(x)  # [B, 256, 60, 80]
        x = self.down4(x)  # [B, 512, 30, 40]
        x = self.down5(x)  # [B, 512, 15, 20]
        
        # 中间层
        x = F.silu(self.mid_conv1(x))
        x = self.mid_attn(x)
        x = F.silu(self.mid_conv2(x))
        
        # 输出均值和方差
        x = self.conv_out(x)  # [B, 16, 15, 20]
        return x

class AdaptiveDecoder(nn.Module):
    """适配480x640输出的解码器"""
    def __init__(self, out_channels=3, latent_channels=8, base_channels=64):
        super().__init__()
        
        # 输入卷积
        self.conv_in = nn.Conv2d(latent_channels, base_channels*8, 3, padding=1)
        self.norm_in = nn.BatchNorm2d(base_channels*8)
        
        # 中间层
        self.mid_conv1 = nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1)
        self.mid_attn = AttentionBlock(base_channels*8)
        self.mid_conv2 = nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1)
        self.norm_mid1 = nn.BatchNorm2d(base_channels*8)
        self.norm_mid2 = nn.BatchNorm2d(base_channels*8)
        
        # 上采样块 - 15x20 -> 30x40 -> 60x80 -> 120x160 -> 240x320 -> 480x640
        self.up1 = UpsampleBlock(base_channels*8, base_channels*8)  # 15x20 -> 30x40
        self.norm_up1 = nn.BatchNorm2d(base_channels*8)
        
        self.up2 = UpsampleBlock(base_channels*8, base_channels*4)  # 30x40 -> 60x80
        self.norm_up2 = nn.BatchNorm2d(base_channels*4)
        
        self.up3 = UpsampleBlock(base_channels*4, base_channels*2)  # 60x80 -> 120x160
        self.norm_up3 = nn.BatchNorm2d(base_channels*2)
        
        self.up4 = UpsampleBlock(base_channels*2, base_channels)    # 120x160 -> 240x320
        self.norm_up4 = nn.BatchNorm2d(base_channels)
        
        self.up5 = FinalUpsampleBlock(base_channels, base_channels) # 240x320 -> 480x640
        self.norm_up5 = nn.BatchNorm2d(base_channels)
        
        # 输出卷积
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, x):
        # 输入: [B, 8, 15, 20]
        x = F.silu(self.norm_in(self.conv_in(x)))
        
        # 中间层
        x = F.silu(self.norm_mid1(self.mid_conv1(x)))
        x = self.mid_attn(x)
        x = F.silu(self.norm_mid2(self.mid_conv2(x)))
        
        # 上采样
        x = F.silu(self.norm_up1(self.up1(x)))  # [B, 512, 30, 40]
        x = F.silu(self.norm_up2(self.up2(x)))  # [B, 256, 60, 80]
        x = F.silu(self.norm_up3(self.up3(x)))  # [B, 128, 120, 160]
        x = F.silu(self.norm_up4(self.up4(x)))  # [B, 64, 240, 320]
        x = F.silu(self.norm_up5(self.up5(x)))  # [B, 64, 480, 640]
        
        # 输出
        x = self.conv_out(x)  # [B, 3, 480, 640]
        return torch.tanh(x)

class FinalUpsampleBlock(nn.Module):
    """最终上采样块 - 精确控制输出尺寸为480x640"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 使用插值精确控制输出尺寸
        self.upsample = nn.Upsample(size=(480, 640), mode='bilinear', align_corners=False)
        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
            
    def forward(self, x):
        residual = self.residual(x)
        
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = x + residual
        
        # 精确上采样到480x640
        x = self.upsample(x)
        
        return x

class DownsampleBlock(nn.Module):
    """下采样块 - 处理480x640尺寸"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 使用卷积下采样而不是池化，更好地控制输出尺寸
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
        
        # 只在通道数较大时使用注意力
        self.attn = AttentionBlock(out_channels) if out_channels >= 256 else nn.Identity()
        
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.AvgPool2d(2)  # 残差路径也需要下采样
            )
        else:
            self.residual = nn.AvgPool2d(2)
            
    def forward(self, x):
        residual = self.residual(x)
        
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = self.downsample(x)  # 下采样
        x = x + residual
        
        x = self.attn(x)
        return x

class UpsampleBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 使用插值上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 只在通道数较大时使用注意力
        self.attn = AttentionBlock(out_channels) if out_channels >= 256 else nn.Identity()
        
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, 1)
            )
        else:
            self.residual = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
    def forward(self, x):
        residual = self.residual(x)
        
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = self.upsample(x)
        x = x + residual
        
        x = self.attn(x)
        return x

class AttentionBlock(nn.Module):
    """自注意力机制"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        # 归一化
        h = self.norm(x)
        
        # 查询、键、值
        q = self.q(h).view(batch, channels, -1).transpose(1, 2)
        k = self.k(h).view(batch, channels, -1)
        v = self.v(h).view(batch, channels, -1).transpose(1, 2)
        
        # 注意力计算
        attention = torch.bmm(q, k) * (channels ** -0.5)
        attention = F.softmax(attention, dim=-1)
        
        # 加权求和
        h = torch.bmm(attention, v)
        h = h.transpose(1, 2).view(batch, channels, height, width)
        
        # 输出投影
        h = self.proj_out(h)
        
        return x + h

class BasicVAE(nn.Module):
    """适配480x640的完整VAE模型"""
    def __init__(self, in_channels=3, latent_channels=8, base_channels=64):
        super().__init__()
        
        self.encoder = AdaptiveEncoder(in_channels, latent_channels, base_channels)
        self.decoder = AdaptiveDecoder(in_channels, latent_channels, base_channels)
        self.latent_channels = latent_channels
        
    def encode(self, x):
        """编码图像到潜在分布参数"""
        h = self.encoder(x)  # [B, 2*latent_channels, 15, 20]
        mean, log_var = torch.chunk(h, 2, dim=1)  # 分割为均值和方差
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        """从潜在变量解码回图像"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        # 编码
        mean, log_var = self.encode(x)
        
        # 重参数化
        z = self.reparameterize(mean, log_var)
        
        # 解码
        reconstructed = self.decode(z)
        
        return reconstructed, mean, log_var
    
    def loss_function(self, reconstructed, original, mean, log_var):
        """VAE损失函数"""
        # 重建损失
        loss=0
        threshold=0.95
        mask=(original<threshold).float()
        
        mask_reverse=1-mask
        error=(reconstructed-original)**2*mask
        loss=error.sum()/mask.sum()
        error1=(reconstructed-original)**2*mask_reverse
        error1=error1.sum()/mask_reverse.sum()
        loss=loss+error1
        # import pdb;pdb.set_trace()
        # for i in range(original.shape[0]):
        #     for j in range(original[i].shape[0]):
        #         for k in range(original[i][j].shape[0]):
        #             for t in range(original[i][j][k].shape[0]):
                        
        #                 if original[i][j][k][t]>=0.95:
        #                     recon_loss = F.mse_loss(reconstructed[i][j][k][t], original[i][j][k][t], reduction='mean')
        #                 else:
        #                     recon_loss=8*F.mse_loss(reconstructed[i][j][k][t], original[i][j][k][t], reduction='mean')
        #                 loss=loss+recon_loss

        # KL散度损失（正则化项）
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        
        # 总损失
        total_loss = loss + 0.001 * kl_loss  # KL权重较小
        return loss
        # return {
        #     'total_loss': total_loss,
        #     'recon_loss': loss,
        #     'kl_loss': kl_loss
        # }
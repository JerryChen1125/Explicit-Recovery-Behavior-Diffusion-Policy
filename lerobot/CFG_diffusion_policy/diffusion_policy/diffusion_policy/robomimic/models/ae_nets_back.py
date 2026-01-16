import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class LightweightConvBlock(nn.Module):
    """轻量级卷积块"""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)

class LightweightAttention(nn.Module):
    """轻量级注意力，类似SD"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_heads, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = [t.reshape(B, self.num_heads, C // self.num_heads, -1) for t in qkv]
        
        # 简化注意力计算
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.einsum('b h c n, b h c m -> b h n m', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('b h n m, b h c m -> b h c n', attn, v)
        out = out.reshape(B, C, H, W)
        
        return residual + self.proj(out)

class LightweightDownsample(nn.Module):
    """轻量级下采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        
    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))

class LightweightUpsample(nn.Module):
    """轻量级上采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return F.silu(self.norm(self.conv(x)))

class CompactEncoder(nn.Module):
    """紧凑编码器 - 专为480x640设计"""
    def __init__(self, in_channels=3, latent_channels=4, base_channels=32):
        super().__init__()
        
        # 输入层
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样路径: 480x640 -> 240x320 -> 120x160 -> 60x80 -> 30x40
        self.down1 = nn.Sequential(
            LightweightConvBlock(base_channels, base_channels),
            LightweightDownsample(base_channels, base_channels*2)  # 64
        )
        
        self.down2 = nn.Sequential(
            LightweightConvBlock(base_channels*2, base_channels*2),
            LightweightDownsample(base_channels*2, base_channels*4)  # 128
        )
        
        self.down3 = nn.Sequential(
            LightweightConvBlock(base_channels*4, base_channels*4),
            LightweightAttention(base_channels*4, num_heads=8),
            LightweightDownsample(base_channels*4, base_channels*8)  # 256
        )
        
        # 中间层
        self.mid = nn.Sequential(
            LightweightConvBlock(base_channels*8, base_channels*8),
            LightweightAttention(base_channels*8, num_heads=8),
            LightweightConvBlock(base_channels*8, base_channels*8),
        )
        
        # 输出层
        self.norm_out = nn.GroupNorm(8, base_channels*8)
        self.conv_out = nn.Conv2d(base_channels*8, latent_channels * 2, 3, padding=1)
        
    def forward(self, x):
        # 输入: [B, 3, 480, 640]
        x = self.conv_in(x)      # [B, 32, 480, 640]
        
        x = self.down1(x)        # [B, 64, 240, 320]
        x = self.down2(x)        # [B, 128, 120, 160]  
        x = self.down3(x)        # [B, 256, 60, 80]
        
        x = self.mid(x)          # [B, 256, 60, 80]
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)     # [B, 8, 60, 80]
        
        mean, log_var = x.chunk(2, dim=1)
        return mean, log_var

class CompactDecoder(nn.Module):
    """紧凑解码器 - 专为480x640设计"""
    def __init__(self, out_channels=3, latent_channels=4, base_channels=32):
        super().__init__()
        
        # 输入层
        self.conv_in = nn.Conv2d(latent_channels, base_channels*8, 3, padding=1)
        
        # 中间层
        self.mid = nn.Sequential(
            LightweightConvBlock(base_channels*8, base_channels*8),
            LightweightAttention(base_channels*8, num_heads=8),
            LightweightConvBlock(base_channels*8, base_channels*8),
        )
        
        # 上采样路径: 30x40 -> 60x80 -> 120x160 -> 240x320 -> 480x640
        self.up1 = nn.Sequential(
            LightweightUpsample(base_channels*8, base_channels*4),  # 256->128
            LightweightConvBlock(base_channels*4, base_channels*4),
            LightweightAttention(base_channels*4, num_heads=8),
        )
        
        self.up2 = nn.Sequential(
            LightweightUpsample(base_channels*4, base_channels*2),  # 128->64
            LightweightConvBlock(base_channels*2, base_channels*2),
        )
        
        self.up3 = nn.Sequential(
            LightweightUpsample(base_channels*2, base_channels),    # 64->32
            LightweightConvBlock(base_channels, base_channels),
        )
        
        # 输出层
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
        
    def forward(self, x, target_size=None):
        # 输入: [B, 4, 60, 80]
        x = self.conv_in(x)      # [B, 256, 60, 80]
        
        x = self.mid(x)          # [B, 256, 60, 80]
        
        x = self.up1(x)          # [B, 128, 120, 160]
        x = self.up2(x)          # [B, 64, 240, 320]
        x = self.up3(x)          # [B, 32, 480, 640]
        
        x = self.conv_out(x)     # [B, 3, 480, 640]
        
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
        return torch.sigmoid(x)

class BasicVAE(nn.Module):
    """轻量级VAE，类似Stable Diffusion设计"""
    def __init__(self, in_channels=3, latent_channels=4, base_channels=32):
        super().__init__()
        
        self.encoder = CompactEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels
        )
        
        self.decoder = CompactDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels, 
            base_channels=base_channels
        )
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x):
        mean, log_var = self.encoder(x)
        return self.reparameterize(mean, log_var), mean, log_var
    
    def decode(self, z, target_size=None):
        return self.decoder(z, target_size)
    
    def forward(self, x):
        z, mean, log_var = self.encode(x)
        recon = self.decode(z, target_size=x.shape[2:])
        return recon, mean, log_var
    def loss_function(self, reconstructed, original, mean, log_var):
        """VAE损失函数"""
        # 重建损失
        
        loss=0
        threshold=0.95
        mask=(original<threshold).float()
        # import pdb;pdb.set_trace()
        mask_reverse=1-mask
        print(reconstructed.shape)
        print(original.shape)
        
        error=(reconstructed-original)**2*mask
        loss=error.sum()/mask.sum()
        error1=(reconstructed-original)**2*mask_reverse
        error1=error1.sum()/mask_reverse.sum()
        loss=loss+error1
        return loss

# 预定义配置
def create_ae_small(in_channels=3, latent_channels=4):
    """小型自编码器"""
    return FlexibleAutoencoder(
        in_channels=in_channels,
        latent_channels=latent_channels,
        base_channels=32,
        encoder_multipliers=[1, 2, 4, 4],
        num_res_blocks=1,
        use_attention=False
    )

def create_ae_medium(in_channels=3, latent_channels=8):
    """中型自编码器"""
    return FlexibleAutoencoder(
        in_channels=in_channels,
        latent_channels=latent_channels,
        base_channels=64,
        encoder_multipliers=[1, 2, 4, 8, 8],
        num_res_blocks=2,
        use_attention=True
    )

def create_ae_large(in_channels=3, latent_channels=16):
    """大型自编码器"""
    return FlexibleAutoencoder(
        in_channels=in_channels,
        latent_channels=latent_channels,
        base_channels=128,
        encoder_multipliers=[1, 2, 4, 8, 16, 16],
        num_res_blocks=3,
        use_attention=True
    )

# 测试函数
def test_flexible_ae():
    """测试自适应自编码器"""
    test_sizes = [
        (2, 3, 84, 84),      # 小尺寸
        (2, 3, 240, 320),    # 中等尺寸  
        (2, 3, 480, 640),    # 标准尺寸
        (2, 3, 108, 192),    # 不规则尺寸
    ]
    
    print("测试不同尺寸的输入:")
    
    # 测试小型自编码器
    print("\n=== 小型自编码器 ===")
    ae_small = create_ae_small()
    
    for i, (b, c, h, w) in enumerate(test_sizes):
        print(f"\n测试尺寸 {h}x{w}:")
        x = torch.randn(b, c, h, w)
        recon, mean, log_var = ae_small(x)
        print(f"输入: {x.shape} -> 重建: {recon.shape}")
        print(f"潜在变量: {mean.shape}, 方差: {log_var.shape}")
    
    # 测试中型自编码器
    print("\n=== 中型自编码器 ===")
    ae_medium = create_ae_medium()
    
    for i, (b, c, h, w) in enumerate(test_sizes):
        print(f"\n测试尺寸 {h}x{w}:")
        x = torch.randn(b, c, h, w)
        recon, mean, log_var = ae_medium(x)
        print(f"输入: {x.shape} -> 重建: {recon.shape}")
        print(f"潜在变量: {mean.shape}, 方差: {log_var.shape}")

if __name__ == "__main__":
    test_flexible_ae()
    
    # 打印模型参数数量
    ae_medium = create_ae_medium()
    total_params = sum(p.numel() for p in ae_medium.parameters())
    print(f"\n中型自编码器参数总数: {total_params:,}")
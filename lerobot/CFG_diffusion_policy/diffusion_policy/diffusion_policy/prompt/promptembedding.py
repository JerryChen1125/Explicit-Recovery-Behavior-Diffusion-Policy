import torch
from torch import Tensor
from typing import Optional
from typing import List

from typing import Type
import torch.nn as nn
import math


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules
# class PromptProjector(nn.Module):
#     def __init__(self, s1,s2,D1,D2):
#         super().__init__()
#         # self.action_dim=D1
#         # self.output_dim=D2
#         # self.state_mlp_activation_fn=nn.ReLU
#         # self.latent_dim=1024
#         # self.state_mlp_size=64
#         # self.net_arch=self.state_mlp_size[:-1]
#         # self.proj = nn.Linear(D1, D2)
#         self.pool = nn.AdaptiveAvgPool1d(s2)  
#         # self.pos_embedding=nn.Embedding(D1,128)# 可以替换为 interpolate 或 attention 模块
#         # self.embedding = ContentAwarePositionEmbeddingSine(self.action_dim,num_pos_feats=64, normalize=True)
#         # self.embedding=PositionEmbeddingSine(D2)
#         # self.embedding=PositionEmbeddingSine(D2)
#         self.s1 = s1
#         self.s2 = s2
#         self.embedding=FourierFeatureMapping(self.action_dim,self.output_dim)
#     def forward(self, x):
#         # x: [b, s1, D1]
#         # import pdb;pdb.set_trace()
#         x=self.embedding(x)

#         # x=self.proj(x)
#         # import pdb;pdb.set_trace()
#         # x=self.pos_embedding(x)
#         # x = self.proj(x)        # [b, s1, D2]
        
#         x = x.permute(0, 2, 1)  # [b, D2, s1] — 方便做池化
#         x = self.pool(x)        # [b, D2, s2]
#         x = x.permute(0, 2, 1)  # [b, s2, D2]
#         return x

class PromptProjector(nn.Module):
    def __init__(self, s1,s2,D1,D2):
        super().__init__()
        self.horizon=s1
        self.obs_steps=s2
        self.action_dim=D1
        self.output_dim=D2
        self.state_mlp_activation_fn=nn.ReLU
        self.latent_dim=1024
        self.state_mlp_size=(64,64)
        self.net_arch=self.state_mlp_size
        self.state_shape =(s1,D1)
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[1], self.output_dim, self.net_arch, self.state_mlp_activation_fn))
        self.mid_dim=self.horizon*self.output_dim//self.obs_steps
        self.last_proj=nn.Linear(self.mid_dim,self.output_dim)
        # self.proj = nn.Linear(D1, D2)
        # self.pool = nn.AdaptiveAvgPool1d(s2)  
        # self.pos_embedding=nn.Embedding(D1,128)# 可以替换为 interpolate 或 attention 模块
        # self.embedding = ContentAwarePositionEmbeddingSine(self.action_dim,num_pos_feats=64, normalize=True)
        # self.embedding=PositionEmbeddingSine(D2)
        # self.embedding=PositionEmbeddingSine(D2)
        # self.s1 = s1
        # self.s2 = s2

    def forward(self, x):
        # x: [b, s1, D1]
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x=self.state_mlp(x)
        batchsize=x.shape[0]

        x=x.reshape(batchsize*self.obs_steps,-1)
        x=self.last_proj(x)
        
        # x=self.proj(x)
        # import pdb;pdb.set_trace()
        # x=self.pos_embedding(x)
        # x = self.proj(x)        # [b, s1, D2]
        
        # x = x.permute(0, 2, 1)  # [b, D2, s1] — 方便做池化
        # x = self.pool(x)        # [b, D2, s2]
        # x = x.permute(0, 2, 1)  # [b, s2, D2]
        return x
    
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on sequences.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        # tensor shape: (b, s, d)
        b, s, d = tensor.shape

        # Create a mask of ones with the same shape as the sequence length
        not_mask = torch.ones(b, s, dtype=torch.float32, device=tensor.device)

        # Compute position embeddings
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # Shape: (b, s)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_y = y_embed[:, :, None] / dim_t  # Shape: (b, s, num_pos_feats)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        # Repeat the position embeddings along the feature dimension
        pos = pos_y.repeat(1, 1, torch.div(d,self.num_pos_feats,rounding_mode='trunc'))  # Shape: (b, s, d)
        if d % self.num_pos_feats != 0:
            pos = torch.cat([pos, torch.zeros(b, s, d % self.num_pos_feats, device=tensor.device)], dim=2)

        return pos
    
class ContentAwarePositionEmbeddingSine(nn.Module):
    def __init__(self, action_dim,num_pos_feats=64, base_temp=10000, normalize=False):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.base_temp = base_temp
        self.normalize = normalize
        # 动态温度生成器：输入特征 -> 温度缩放因子
        self.temp_proj = nn.Sequential(
            nn.Linear(action_dim, num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, 1),
            nn.Sigmoid()  # 输出在 (0, 1) 范围内
        )

    def forward(self, x):
        b, s, d = x.shape
        not_mask = torch.ones(b, s, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:] + 1e-6) * 2 * math.pi

        # 动态温度：基于输入特征计算
        # import pdb;pdb.set_trace()
        temp_scale = self.temp_proj(x.mean(dim=1))  # [b, 1]
        temperature = self.base_temp * temp_scale  # 每批次样本有不同的温度

        dim_t = torch.arange(self.num_pos_feats, device=x.device)
        dim_t = temperature.view(-1, 1, 1) ** (2 * torch.div(dim_t ,2 ,rounding_mode='trunc')/ self.num_pos_feats)

        pos_y = y_embed.unsqueeze(2) / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        # 重复以匹配维度
        pos = pos_y.repeat(1, 1, torch.div(d, self.num_pos_feats,rounding_mode='trunc'))
        if d % self.num_pos_feats != 0:
            pos = torch.cat([pos, torch.zeros(b, s, d % self.num_pos_feats, device=x.device)], dim=2)
        
        return pos
    
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
    
# class PositionEmbeddingSine(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on images.
#     """
#     def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale

#     def forward(self, tensor):
#         x = tensor
#         # mask = tensor_list.mask
#         # assert mask is not None
#         # not_mask = ~mask

#         not_mask = torch.ones_like(x[0, [0]])
#         y_embed = not_mask.cumsum(1, dtype=torch.float32)
#         x_embed = not_mask.cumsum(2, dtype=torch.float32)
#         if self.normalize:
#             eps = 1e-6
#             y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
#             x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

#         dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

#         pos_x = x_embed[:, :, :, None] / dim_t
#         pos_y = y_embed[:, :, :, None] / dim_t
#         pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#         return pos
    
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=0.015):
        """
        Fourier feature mapping module.
        
        Args:
            input_dim: Dimension of input data
            mapping_size: Output dimension after mapping
            scale: Scaling factor for the Fourier features
        """
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        
        # Random Fourier feature matrix (B in the paper)
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size // 2) , 
                             requires_grad=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim)
        Returns:
            Mapped features of shape (..., mapping_size)
        """
        # Project input onto Fourier basis
        # import pdb;pdb.set_trace()
        proj =  math.pi * x @ self.B  # (..., mapping_size//2)
        
        # Concatenate sin and cos features
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # import pdb;pdb.set_trace()
        x = x.unsqueeze(-1) 
        emb = x * emb[None,None, :]
        emb=emb.sum(dim=-2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
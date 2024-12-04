import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange, repeat

from torch.cuda.amp import custom_bwd, custom_fwd
from timm.models.layers import DropPath, to_2tuple, to_3tuple, make_divisible, trunc_normal_
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))
    
# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class PatchPartition(nn.Module):
    def __init__(self, channels):
        super(PatchPartition, self).__init__()
        self.positional_encoding = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels,
                                                         bias=False)

    def forward(self, x):
        x = self.positional_encoding(x)
        return x

class Mlp(nn.Module):
    def __init__(self, channels, shallow=True):
        super(Mlp, self).__init__()
        expansion = 4
        self.line_conv_0 = nn.Conv3d(channels, channels * expansion, kernel_size=1, bias=False)
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.line_conv_1 = nn.Conv3d(channels * expansion, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.line_conv_0(x)
        x = self.act(x)
        x = self.line_conv_1(x)
        return x

class LocalRepresentationsCongregation(nn.Module):
    def __init__(self, channels):
        super(LocalRepresentationsCongregation, self).__init__()
        self.bn1 = nn.BatchNorm3d(channels)
        self.pointwise_conv_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv3d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.pointwise_conv_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba_type="v3",
                # nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class GlobalSparseTransformer(nn.Module):
    def __init__(self, channels, r, heads, num_slices=64):
        super(GlobalSparseTransformer, self).__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
        
        self.mamba = MambaLayer(channels, num_slices=num_slices)
        # qkv
        # self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        identity = x
        
        # Mamba
        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D).contiguous() 
        x = x + identity
        
        return x


class LocalReverseDiffusion(nn.Module):
    def __init__(self, channels, r):
        super(LocalReverseDiffusion, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.conv_trans = nn.ConvTranspose3d(channels,
                                             channels,
                                             kernel_size=r,
                                             stride=r,
                                             groups=channels)
        self.pointwise_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.norm(x)
        x = self.pointwise_conv(x)
        return x



class Block(nn.Module):
    def __init__(self, channels, r, heads, num_slices, shallow=True):
        super(Block, self).__init__()

        self.patch1 = PatchPartition(channels)
        self.LocalRC = LocalRepresentationsCongregation(channels)
        self.LineConv1 = Mlp(channels, shallow)
        self.patch2 = PatchPartition(channels)
        
        # self.GlobalST = GlobalSparseTransformer(channels, r, heads, num_slices)
        # self.LocalRD = LocalReverseDiffusion(channels, r)
        
        self.mamba = MambaLayer(channels, num_slices=num_slices)
        
        self.LineConv2 = Mlp(channels, shallow)

    def forward(self, x):
        x = self.patch1(x) + x
        x = self.LocalRC(x) + x
        x = self.LineConv1(x) + x
        x = self.patch2(x) + x
        
        # x = self.LocalRD(self.GlobalST(x)) +x
        
        int_x = x
        # Mamba
        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D).contiguous() 
        
        x = x + int_x
        
        x = self.LineConv2(x) + x
        return x

class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DepthwiseConvLayer, self).__init__()
        self.depth_wise = nn.Conv3d(dim_in,
                                    dim_out,
                                    kernel_size=r,
                                    stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.norm(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=4, embed_dim=384, embedding_dim=27, channels=(48, 96, 240),
                 blocks=(1, 2, 3, 2), heads=(1, 2, 4, 8), r=(4, 2, 2, 2), num_slices_list = [64, 32, 16, 8], distillation=False, dropout=0.3):
        super(Encoder, self).__init__()
        self.distillation = distillation
        self.DWconv1 = DepthwiseConvLayer(dim_in=in_channels, dim_out=channels[0], r=r[0])
        self.DWconv2 = DepthwiseConvLayer(dim_in=channels[0], dim_out=channels[1], r=r[1])
        self.DWconv3 = DepthwiseConvLayer(dim_in=channels[1], dim_out=channels[2], r=r[2])
        self.DWconv4 = DepthwiseConvLayer(dim_in=channels[2], dim_out=embed_dim, r=r[3])
        block = []
        for _ in range(blocks[0]):
            block.append(Block(channels=channels[0], r=r[0], heads=heads[0], num_slices=num_slices_list[0], shallow=True))
        self.block1 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[1]):
            block.append(Block(channels=channels[1], r=r[1], heads=heads[1], num_slices=num_slices_list[1], shallow=True))
        self.block2 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2], num_slices=num_slices_list[2], shallow=False))
        self.block3 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3], num_slices=num_slices_list[3], shallow=False))
        self.block4 = nn.Sequential(*block)
        self.position_embeddings = nn.Parameter(torch.zeros(1, embedding_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden_states_out = []
        x = self.DWconv1(x)
        x = self.block1(x)
        hidden_states_out.append(x)
        x = self.DWconv2(x)
        x = self.block2(x)
        hidden_states_out.append(x)
        x = self.DWconv3(x)
        x = self.block3(x)
        hidden_states_out.append(x)
        x = self.DWconv4(x)
        B, C, W, H, Z = x.shape
        x = self.block4(x)
        x = x.flatten(2).transpose(-1, -2)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x, hidden_states_out, (B, C, W, H, Z)

class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(TransposedConvLayer, self).__init__()
        self.transposed = nn.ConvTranspose3d(dim_in,
                                             dim_out,
                                             kernel_size=r,
                                             stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.transposed(x)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=3, embed_dim=384, channels=(48, 96, 240), num_slices_list = [64, 32, 16, 8],
                 blocks=(1, 2, 3, 2), heads=(1, 2, 4, 8), r=(4, 2, 2, 2), distillation=False, dropout=0.3):
        super(Decoder, self).__init__()
        self.distillation = distillation

        self.SegHead = TransposedConvLayer(dim_in=channels[0], dim_out=out_channels, r=r[0])
        self.TSconv3 = TransposedConvLayer(dim_in=channels[1], dim_out=channels[0], r=r[1])
        self.TSconv2 = TransposedConvLayer(dim_in=channels[2], dim_out=channels[1], r=r[2])
        self.TSconv1 = TransposedConvLayer(dim_in=embed_dim, dim_out=channels[2], r=r[3])

        block = []
        for _ in range(blocks[0]):
            block.append(Block(channels=channels[0], r=r[0], heads=heads[0], num_slices=num_slices_list[0], shallow=True))
        self.block1 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[1]):
            block.append(Block(channels=channels[1], r=r[1], heads=heads[1], num_slices=num_slices_list[1], shallow=True))
        self.block2 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2], num_slices=num_slices_list[2], shallow=False))
        self.block3 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3], num_slices=num_slices_list[3], shallow=False))
        self.block4 = nn.Sequential(*block)
        
        # self.fblock4 = FeatureBlock(in_dim=channels[3], num_slices=num_slices_list[0], shallow=True)
        self.fblock3 = FeatureBlock(in_dim=channels[2], num_slices=num_slices_list[3], shallow=True)
        self.fblock2 = FeatureBlock(in_dim=channels[1], num_slices=num_slices_list[2], shallow=False)
        self.fblock1 = FeatureBlock(in_dim=channels[0], num_slices=num_slices_list[1], shallow=False)

    def forward(self, x, hidden_states_out, x_shape):
        B, C, W, H, Z = x_shape
        x = x.reshape(B, C, W, H, Z)
        x = self.block4(x)
        x = self.TSconv1(x)
        x = x + self.fblock3(hidden_states_out[2])
        x = x + hidden_states_out[2]
        x = self.block3(x)
        x = self.TSconv2(x)
        x = x + self.fblock2(hidden_states_out[1])
        x = x + hidden_states_out[1]
        x = self.block2(x)
        x = self.TSconv3(x)
        x = x + self.fblock1(hidden_states_out[0])
        x = x + hidden_states_out[0]
        x = self.block1(x)
        x = self.SegHead(x)
        return x
        # x = x.reshape(B, C, W, H, Z)
        # x = self.block4(x)
        # x = self.TSconv1(x)
        # # x = x + self.fblock3(hidden_states_out[2])
        # x = x + hidden_states_out[2]
        # x = self.block3(x)
        # x = self.TSconv2(x)
        # # x = x + self.fblock2(hidden_states_out[1])
        # x = x + hidden_states_out[1]
        # x = self.block2(x)
        # x = self.TSconv3(x)
        # # x = x + self.fblock1(hidden_states_out[0])
        # x = x + hidden_states_out[0]
        # x = self.block1(x)
        # x = self.SegHead(x)
        # return x

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, act=nn.ReLU(inplace=True)):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class GobleAttention(nn.Module):
    def __init__(self, in_dim=1, out_dim=32, num_slices=8, kernel_size=3, shallow=True):
        super(GobleAttention, self).__init__()
        # 调整通道Conv
        self.conv = nn.Conv3d(in_dim, out_dim, 3, 1, 1)   
        # 特征Norm
        self.norm = nn.GroupNorm(out_dim // 2, out_dim)
        # 激活函数
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        
        self.base_norm = nn.BatchNorm3d(out_dim)
        
        self.base_conv = nn.Conv3d(out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, 1, out_dim, bias=False)
        self.base_norm = nn.BatchNorm3d(out_dim)
        
        self.add_conv = nn.Conv3d(out_dim, out_dim, 1, 1, 0, 1, out_dim, bias=False)
        self.add_norm = nn.BatchNorm3d(out_dim)
        
        # Mamba
        # self.mamba = MambaLayer(out_dim, num_slices=num_slices)
        
        # MLP
        self.mlp = Mlp(out_dim, shallow)
        
        # self.apply(self._init_weights) 
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            # 计算fan_out，注意Conv3d的情况下kernel_size是一个三元组
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            # 使用标准正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果存在偏置，则将其初始化为0
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, x):
        # 调整通道
        x = self.conv(x)
        # 特征Norm
        x = self.norm(x)
        # 特征激活
        x = self.act(x)
        # keep input
        identity = x
        
        x = self.base_norm(self.base_conv(x)) + self.add_norm(self.add_conv(x)) + x
        
        # # Mamba
        # B, C, H, W, D = x.shape
        # x = x.flatten(2).transpose(1, 2)
        # x = x.permute(0, 2, 1)
        # x = self.mamba(x)
        # x = x.permute(0, 2, 1)
        # x = x.permute(0, 2, 1).reshape(B, C, H, W, D).contiguous() 
        # x = x + identity
        
        # MLP
        x = self.mlp(x)
        
        return x + identity

class LocalAttention(nn.Module):
    def __init__(self, in_dim=32, out_dim=32):
        super(LocalAttention, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_dim)
        self.pointwise_conv_0 = nn.Conv3d(in_dim, in_dim, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv3d(in_dim, in_dim, padding=1, kernel_size=3, groups=in_dim, bias=False)
        self.bn2 = nn.BatchNorm3d(in_dim)
        self.pointwise_conv_1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, bias=False)
        # self.apply(self._init_weights) 
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            # 计算fan_out，注意Conv3d的情况下kernel_size是一个三元组
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            # 使用标准正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果存在偏置，则将其初始化为0
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, act=nn.ReLU(inplace=True)):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class FeatureBlock(nn.Module):
    def __init__(self, in_dim=3, num_slices=8, shallow=True):
        super(FeatureBlock, self).__init__()
        self.in_dim = in_dim
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.gobel_attention = GobleAttention(in_dim=in_dim//2, out_dim=in_dim, num_slices=num_slices, shallow=shallow)
        self.local_attention = LocalAttention(in_dim=in_dim//2, out_dim=in_dim)
        self.downsample = BasicConv3d(in_dim*2, in_dim, 1, act=self.act)

    def forward(self, x):
        x_0, x_1 = x.chunk(2,dim = 1)
        x_0 = self.gobel_attention(x_0)
        x_1 = self.local_attention(x_1)
        x = torch.cat([x_0, x_1], dim=1)
        x = self.downsample(x)
        return x


class SlimUNETR(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, embed_dim=96,embedding_dim=8, channels=(24, 48, 60),
                        blocks=(1, 1, 1, 1), heads=(1, 2, 4, 4), r=(4, 4, 2, 2), num_slices_list = (64, 32, 16, 8), distillation=False,
                        dropout=0.3):
        super(SlimUNETR, self).__init__()
        self.Encoder = Encoder(in_channels=in_channels, embed_dim=embed_dim,
                                                   embedding_dim=embedding_dim,
                                                   channels=channels,
                                                   blocks=blocks, heads=heads, r=r, distillation=distillation,num_slices_list=num_slices_list,
                                                   dropout=dropout)
        self.Decoder = Decoder(out_channels=out_channels, embed_dim=embed_dim, channels=channels,num_slices_list=num_slices_list,
                                      blocks=blocks, heads=heads, r=r, distillation=distillation, dropout=dropout)

    def forward(self, x):
        embeding, hidden_states_out, (B, C, W, H, Z) = self.Encoder(x)
        x = self.Decoder(embeding, hidden_states_out, (B, C, W, H, Z))
        return x
    
def test_weight(model, x):
    for i in range(0, 5):
        _ = model(x)
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(flops, params, throughout):
    print('params : {} M'.format(round(params / (1000**2), 2)))
    print('flop : {} G'.format(round(flops / (1000**3), 2)))
    print('throughout: {} FPS'.format(throughout))

if __name__ == '__main__':
    device = 'cuda:2'
    x = torch.randn(size=(1, 4, 128, 128, 128)).to(device)
    model = SlimUNETR().to(device)
    print(model(x).shape)
    flops, param, throughout = test_weight(model, x)
    Unitconversion(flops, param, throughout)
    
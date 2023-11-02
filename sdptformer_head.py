# ---------------------------------------------------------------
# Copyright (c) 2021,  Corporation. All rights reserved.
#
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

from timm.models.layers import trunc_normal_
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = pool
        self.lf = DWConv(dim)
        self.sigmoid = h_sigmoid()

        if pool:
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
            self.softmax = nn.Softmax(-1)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        identity = x
        B, N, C = x.shape
        q_ = self.q(x)
        q = q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.pool:
            x_ = x.permute(0,2,1).contiguous().reshape(B,C,H,W)
            x_h = self.pool_h(x_) #B C H 1  
            x_w = self.pool_w(x_) #B C 1 W
            x_w = x_w.permute(0,1,3,2).contiguous() 
            x_ = torch.cat([x_h,x_w],dim=2)  #B C H+W 1 
            x_ = x_.reshape(B,C,-1).permute(0,2,1).contiguous() #B H+W C  
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.sigmoid(x) * self.lf(q_, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = F.relu(x + identity, inplace=True)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # b n c -> b c h w

        return x

@HEADS.register_module()
class SdptformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.sdpatt = Attention(self.channels,num_heads=2)

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        _c2, _c3, _c4 = outs
        c = sum(outs) / len(outs)
        _, _, H, W = c.shape
        c = c.flatten(2).transpose(1, 2).contiguous() # b n c
        f = self.sdpatt(c, H, W) # b c h w
        _c2 = _c2 + f
        _c3 = _c3 + f
        _c4 = _c4 + f
        _c = torch.cat([_c4, _c3, _c2], dim=1)

        out = self.fusion_conv(_c)

        out = self.cls_seg(out)

        return out

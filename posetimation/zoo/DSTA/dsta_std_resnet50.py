#!/usr/bin/python
# -*- coding:utf8 -*-
import math
import os

import numpy as np
import torch
import torch.nn as nn
import logging

from easydict import EasyDict
from einops import rearrange
from ..base import BaseModel

from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY
from timm.models.layers import DropPath
from functools import partial
import torch.distributions as distributions
from mmpose.models.heads.rle_regression_head import nets, nett, RealNVP
from mmpose.core.evaluation import keypoint_pck_accuracy
from ..backbones.Resnet import ResNet


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y

@MODEL_REGISTRY.register()
class DSTA_STD_ResNet50(BaseModel):

    def __init__(self, cfg, phase, **kwargs):
        super(DSTA_STD_ResNet50, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
        self.is_train = True if phase == TRAIN_PHASE else False
        self.freeze_hrnet_weights = cfg.MODEL.FREEZE_HRNET_WEIGHTS
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.preact = ResNet('resnet50')

        self.pretrained = cfg.MODEL.PRETRAINED
        self.embed_dim_ratio = 32
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(3, self.embed_dim_ratio))
        drop_path_rate = 0.1
        depth = 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # embed_dim = self.embed_dim_ratio * 17
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim_ratio, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim_ratio, num_heads=8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.Spatial_norm = norm_layer(self.embed_dim_ratio)
        self.Temporal_norm = norm_layer(self.embed_dim_ratio)
        self.num_frame = 3
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame, self.embed_dim_ratio))
        self.weighted_mean = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.head_coord = nn.Sequential(
            nn.LayerNorm(self.embed_dim_ratio),
            nn.Linear(self.embed_dim_ratio, 2),
        )
        self.head_sigma = nn.Sequential(
            nn.LayerNorm(self.embed_dim_ratio),
            nn.Linear(self.embed_dim_ratio, 2),
        )

        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        prior = distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
        self.flow = RealNVP(nets, nett, masks, prior)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_coord = Linear(2048, self.num_joints * self.embed_dim_ratio)
        self.joint_ordor =  [1,2,0,0,0,3,6,4,7,5,8,9,12,10,13,11,14]


    def forward(self, x, meta=None):
        batch_size = x.shape[0]
        x = torch.cat(x.split(3, dim=1), 0)
        feat = self.preact(x)
        feat = self.avg_pool(feat).reshape(batch_size * self.num_frame, -1)
        feat = self.fc_coord(feat).reshape(batch_size * self.num_frame, self.num_joints, self.embed_dim_ratio)
        feat = torch.stack(feat.split(batch_size, dim=0), dim=0)

        #SD
        feat_S = feat[self.num_frame // 2]
        unused_keypoint_token = feat_S[:, -2:]  # shape : bs, 2, dims
        feat_S = feat_S[:, :15]
        feat_S = rearrange(feat_S, 'b (g n) c  -> b g n c', g=5)
        feat_S = feat_S + self.Spatial_pos_embed
        feat_S = rearrange(feat_S, 'b g n c -> (b g) n c')
        for blk in self.Spatial_blocks:
            feat_S = blk(feat_S)  
        feat_S = self.Spatial_norm(feat_S)
        feat_S = rearrange(feat_S, '(b g) n c  -> b g n c', g=5)
        feat_S = rearrange(feat_S, 'b g n c ->b (g n) c')
        feat_S = torch.concat([feat_S, unused_keypoint_token], dim=1)
        

        #TD
        feat_T = rearrange(feat, 'f b p c -> (b p) f c')
        feat_T = feat_T + self.Temporal_pos_embed
        for blk in self.blocks:
            feat_T = blk(feat_T)
        feat_T = self.Temporal_norm(feat_T)
        feat_T = rearrange(feat_T, '(b p) f c -> f b p c', p=self.num_joints)
        feat_T = feat_T[self.num_frame // 2]

        #Aggregation
        feat = torch.stack((feat_T, feat_S), dim=1)
        feat = self.weighted_mean(feat).squeeze()
        idx = self.joint_ordor
        feat = feat[:, idx]

        coord = self.head_coord(feat).reshape(batch_size, self.num_joints, 2)
        sigma = self.head_sigma(feat).reshape(batch_size, self.num_joints, 2).sigmoid()
        score = 1 - sigma
        score = torch.mean(score, dim=2, keepdim=True)
        if self.is_train:
            target = meta['target'].cuda()
            target_weight = meta['target_weight'].cuda()
            bar_mu = (coord - target) / sigma
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(batch_size, self.num_joints, 1)
            nf_loss = torch.log(sigma) - log_phi

            acc = self.get_accuracy(coord, target, target_weight)
        else:
            nf_loss = None
            acc = None

        output = EasyDict(
            pred_jts=coord,
            sigma=sigma,
            maxvals=score.float(),
            nf_loss=nf_loss,
            acc=acc
        )
        return output


    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        N = output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        return avg_acc

    def init_weights(self):
        logger = logging.getLogger(__name__)
        ## init_weights
        preact_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net 单独判断一下
            if module_name.split('.')[0] == "preact":
                preact_name_set.add(module_name)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        if os.path.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] == '*':
                    layer_name = name.split('.')[0]
                    if layer_name in preact_name_set:
                        need_init_state_dict[name] = m
                    else:
                        new_layer_name = "preact.{}".format(layer_name)
                        if new_layer_name in preact_name_set:
                            parameter_name = "preact.{}".format(name)
                            need_init_state_dict[parameter_name] = m
            # TODO pretrained from posewarper not test
            self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(self.pretrained))


    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):
        # dilation = cfg.MODEL.DEFORMABLE_CONV.DILATION
        # dilation_str = ",".join(map(str, dilation))
        # hyper_parameters_setting = "D_{}".format(dilation_str)
        return 'Resnet50'

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = DSTA_STD_ResNet50(cfg, phase, **kwargs)
        return model

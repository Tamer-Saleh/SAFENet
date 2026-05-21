import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from .Base_model import pvt_v2_b2
import math

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pvt = pvt_v2_b2()
        pvt_path = './PRETRAINED/pvt_v2_b2.pth'
        pvt_save_model = torch.load(pvt_path)
        pvt_model_dict = self.pvt.state_dict()
        pvt_state_dict = {k: v for k, v in pvt_save_model.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(pvt_state_dict)
        self.pvt.load_state_dict(pvt_model_dict)

    def forward(self, A, B):
        pvta = self.pvt(A)
        pvtb = self.pvt(B)
        return pvta, pvtb
    

class F_Enhance(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_size):
        super(F_Enhance, self).__init__()
        self.target_size = target_size
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in in_channels_list
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        processed = []
        for i, conv in enumerate(self.conv_layers):
            feat = conv(features[i])
            if feat.size(2) != self.target_size:
                feat = nn.functional.interpolate(feat, size=self.target_size, mode='bilinear', align_corners=True)
            processed.append(feat)
        fused = torch.cat(processed, dim=1) 
        out = self.fuse_conv(fused)
        return out

class EdgeExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.edge(x)

class CrossNeighborConsistency(nn.Module):
    def __init__(self, channels_low, channels_high):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels_low + channels_high, channels_low, 3, padding=1),
            nn.BatchNorm2d(channels_low),
            nn.Sigmoid()
        )

    def forward(self, edge_low, edge_high):
        edge_high = F.interpolate(
            edge_high,
            size=edge_low.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        consistency = self.fuse(
            torch.cat([edge_low, edge_high], dim=1)
        )
        return consistency


class CNEC(nn.Module):
    def __init__(self):
        super(CNEC, self).__init__()

        self.feature_enhance = F_Enhance(
            in_channels_list=[64, 128, 320, 512],
            out_channels=128,
            target_size=64
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64,  kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64,  128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(320, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.edge1 = EdgeExtractor(64)
        self.edge2 = EdgeExtractor(128)
        self.edge3 = EdgeExtractor(320)
        self.edge4 = EdgeExtractor(512)

        self.cnc_12 = CrossNeighborConsistency(channels_low=64,  channels_high=128)
        self.cnc_23 = CrossNeighborConsistency(channels_low=128, channels_high=320)
        self.cnc_34 = CrossNeighborConsistency(channels_low=320, channels_high=512)

    def forward(self, pvt):
        pvt_1, pvt_2, pvt_3, pvt_4 = pvt

        enhanced = self.feature_enhance([pvt_1, pvt_2, pvt_3, pvt_4]) 

        dec1 = self.decoder1(enhanced)
        dec1_fused = dec1 + pvt_1                                       

        dec2 = self.decoder2(dec1_fused)
        dec2 = F.interpolate(dec2, size=pvt_2.shape[2:], mode='bilinear', align_corners=True)
        dec2_fused = dec2 + pvt_2                                       

        dec3 = self.decoder3(dec2_fused)
        dec3 = F.interpolate(dec3, size=pvt_3.shape[2:], mode='bilinear', align_corners=True)
        dec3_fused = dec3 + pvt_3                                       

        dec4 = self.decoder4(dec3_fused)
        dec4 = F.interpolate(dec4, size=pvt_4.shape[2:], mode='bilinear', align_corners=True)
        dec4_fused = dec4 + pvt_4                                       

        e1 = self.edge1(dec1_fused)   
        e2 = self.edge2(dec2_fused)   
        e3 = self.edge3(dec3_fused)  
        e4 = self.edge4(dec4_fused)   

        g12 = self.cnc_12(e1, e2)
        dec1_fused = dec1_fused + dec1_fused * g12

        g23 = self.cnc_23(e2, e3)
        dec2_fused = dec2_fused + dec2_fused * g23

        g34 = self.cnc_34(e3, e4)
        dec3_fused = dec3_fused + dec3_fused * g34

        return dec1_fused, dec2_fused, dec3_fused, dec4_fused

        
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
        
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


class RGME(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size, activation='relu'):
        super(RGME, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
            act_layer(activation, inplace=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
            act_layer(activation, inplace=True)
        )
        
        self.depthwise_3 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=3, padding=1, groups=F_int, bias=False),
            nn.BatchNorm2d(F_int),
            act_layer(activation, inplace=True)
        )
        
        self.depthwise_5 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=5, padding=2, groups=F_int, bias=False),
            nn.BatchNorm2d(F_int),
            act_layer(activation, inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, bias=False),
            act_layer(activation, inplace=True)
        ) 
        
        self.psi_fusion = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.final_gate = nn.Sequential(
            nn.Conv2d(F_int, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        fused = g1 + x1

        c3 = self.depthwise_3(fused)
        c5 = self.depthwise_5(fused)
        
        g_pool = self.global_pool(fused)
        g_feat = self.global_fc(g_pool)
        g_feat = F.interpolate(g_feat, size=fused.shape[2:], mode='nearest')

        fused_ms = c3 + c5 + g_feat
        attn_map = self.psi_fusion(fused_ms)
        final_gate_weights = self.final_gate(attn_map)
        
        return x * final_gate_weights + x
    

class LPCU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, activation='relu'):
        super(LPCU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        target_in_channels = out_channels * (scale_factor ** 2)
        
        self.prep_pwc = nn.Sequential(
            nn.Conv2d(in_channels, target_in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_in_channels),
            act_layer(activation, inplace=True)
        )
        
        self.upsample = nn.PixelShuffle(scale_factor)
        
        self.residual_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(activation, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.final_activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x_prep = self.prep_pwc(x)
        x_up = self.upsample(x_prep)
        x_res = self.residual_block(x_up)
        out = self.final_activation(x_up + x_res)
        return out
    

class GatedFeatureBlock(nn.Module):
    def __init__(self, dim, expansion_factor=2, dropout=0.):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gate = self.act(self.gate_proj(x))
        value = self.value_proj(x)
        out = gate * value
        out = self.out_proj(out)
        return self.dropout(out)


class LLPF(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            groups=channels,   
            bias=False,
        )
        self._init_gaussian()

    def _init_gaussian(self):
        k = self.kernel_size
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8 
        coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g2d = torch.ger(g1d, g1d)
        g2d = g2d / g2d.sum() 
        with torch.no_grad():
            self.dw_conv.weight.copy_(
                g2d.unsqueeze(0).unsqueeze(0)
                   .expand(self.dw_conv.weight.shape)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)


class AFF(nn.Module):
    def __init__(self, hi_dim: int, lo_dim: int, reduction: int = 4):
        super().__init__()
        total = hi_dim + lo_dim
        hidden = max(total // reduction, 8)

        self.gate = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total),
            nn.Sigmoid(),           
        )

    def forward(self, f_hi: torch.Tensor, f_lo: torch.Tensor) -> torch.Tensor:
        f_cat = torch.cat([f_hi, f_lo], dim=-1)       

        gap = f_cat.mean(dim=1)                         
        g = self.gate(gap).unsqueeze(1)                
        return g * f_cat + (1.0 - g) * f_cat          


class AFFScaled(nn.Module):
    def __init__(self, hi_dim: int, lo_dim: int, reduction: int = 4):
        super().__init__()
        total = hi_dim + lo_dim
        hidden = max(total // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total),
            nn.Sigmoid(),
        )

    def forward(self, f_hi: torch.Tensor, f_lo: torch.Tensor) -> torch.Tensor:
        f_cat = torch.cat([f_hi, f_lo], dim=-1)       
        gap   = f_cat.mean(dim=1)                      
        g     = self.gate(gap).unsqueeze(1)             
        return f_cat * g                                


class HLFA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
        qk_scale=None, attn_drop=0., proj_drop=0., window_size=7, alpha=0.5,
        ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim  = dim
        self.l_heads = int(num_heads * alpha)
        self.l_dim   = self.l_heads * head_dim
        self.h_heads = num_heads - self.l_heads
        self.h_dim   = self.h_heads * head_dim
        self.ws      = window_size

        if self.ws == 1:
            self.h_heads = 0
            self.h_dim   = 0
            self.l_heads = num_heads
            self.l_dim   = dim

        self.init_submodules(head_dim, qkv_bias, qk_scale, attn_drop, proj_drop)

    def init_submodules(self, head_dim, qkv_bias, qk_scale, attn_drop, proj_drop):
        self.scale = qk_scale or head_dim ** -0.5

        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = LLPF(channels=self.dim, kernel_size=self.ws)

            self.l_q  = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Sequential(
                nn.Linear(self.l_dim, self.l_dim),
                nn.Dropout(proj_drop),
            )

        if self.h_heads > 0:
            self.h_qkv  = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = GatedFeatureBlock(self.h_dim, dropout=proj_drop)

        if self.h_heads > 0 and self.l_heads > 0:
            self.aff = AFFScaled(
                hi_dim=self.h_dim,
                lo_dim=self.l_dim,
            )

    def Lo_phi(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        x_window = (
            x.reshape(B, h_group, self.ws, w_group, self.ws, C)
             .transpose(2, 3)
        )

        qkv = (
            self.h_qkv(x_window)
                .reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads)
                .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = (
            (attn @ v)
            .transpose(2, 3)
            .reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        )

        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
        x = self.h_proj(x)
        return x.reshape(B, N, self.h_dim)

    def Hi_phi(self, x, H, W):
        B, N, C = x.shape
        x_map = x.reshape(B, H, W, C)

        q = (
            self.l_q(x)
                .reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads)
                .permute(0, 2, 1, 3)
        )

        if self.ws > 1:
            x_ = x_map.permute(0, 3, 1, 2)          
            x_ = self.sr(x_)                          
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1) 
            kv = (
                self.l_kv(x_)
                    .reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads)
                    .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.l_kv(x)
                    .reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads)
                    .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x.reshape(B, N, self.l_dim)

    def forward(self, x_map):
        B, C, H, W = x_map.shape
        N = H * W
        x = x_map.permute(0, 2, 3, 1).reshape(B, N, C)

        if self.h_heads == 0:
            x = self.Hi_phi(x, H, W)
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.l_heads == 0:
            x = self.Lo_phi(x, H, W)
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        Lo_phi_out = self.Lo_phi(x, H, W) 
        Hi_phi_out = self.Hi_phi(x, H, W)  

        x_out = self.aff(Lo_phi_out, Hi_phi_out) 

        return x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)


def movement(y, n=5):
    B, C, H, W = y.shape
    num = C // n
    out = torch.zeros_like(y)
    out[:, num * 0:num * 1, 1:, :] = y[:, num * 0:num * 1, :-1, :] 
    out[:, num * 1:num * 2, :-1, :] = y[:, num * 1:num * 2, 1:, :] 
    out[:, num * 2:num * 3, :, :-1] = y[:, num * 2:num * 3, :, 1:] 
    out[:, num * 3:num * 4, :, 1:] = y[:, num * 3:num * 4, :, :-1]
    out[:, num * 4:, :, :] = y[:, num * 4:, :, :] 
    return out


class CSC(nn.Module):
    def __init__(self, in_channels, n=6):
        super(CSC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1) 
        self.movement_fn = partial(movement, n=n)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.movement_fn(x)
        x = self.conv2(x)
        return x


class SGU(nn.Module):
    def __init__(self, dim):
        super(SGU, self).__init__()
        expansion_factor = 4
        hidden_dim = dim * expansion_factor
        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1) 
        self.norm1 = nn.LayerNorm(dim) 
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.project_out = nn.Conv2d(hidden_dim // 2, dim, kernel_size=1)

    def forward(self, x):
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        x_exp = self.project_in(x_norm)
        x_dwconv = self.dwconv(x_exp)
        
        x1, x2 = x_dwconv.chunk(2, dim=1)
        x_gated = F.gelu(x1) * x2
        
        out = self.project_out(x_gated)
        return x + out


class SGHiLo_Phi(nn.Module):
    def __init__(self, channels, opt, lgag_ks=3):
        super(SGHiLo_Phi, self).__init__()
        self.opt = opt
        C4, C3, C2, C1 = channels[0], channels[1], channels[2], channels[3]
        
        self.csc4 = CSC(C4)
        self.HLFA4 = HLFA(dim=C4, num_heads=8, window_size=4)

        self.LPCU3 = LPCU(in_channels=C4, out_channels=C3)
        self.RGME3 = RGME(F_g=C3, F_l=C3, F_int=C3 // 2, kernel_size=lgag_ks)
        self.sgu3 = SGU(C3)
        self.csc3 = CSC(C3)
        self.HLFA3 = HLFA(dim=C3, num_heads=8, window_size=8)

        self.LPCU2 = LPCU(in_channels=C3, out_channels=C2)
        self.RGME2 = RGME(F_g=C2, F_l=C2, F_int=C2 // 2, kernel_size=lgag_ks)
        self.sgu2 = SGU(C2) 
        self.csc2 = CSC(C2)
        self.HLFA2 = HLFA(dim=C2, num_heads=8, window_size=8)

        self.LPCU1 = LPCU(in_channels=C2, out_channels=C1)
        self.RGME1 = RGME(F_g=C1, F_l=C1, F_int=C1 // 2, kernel_size=lgag_ks)
        self.sgu1 = SGU(C1) 
        self.csc1 = CSC(C1)
        self.HLFA1 = HLFA(dim=C1, num_heads=8, window_size=8)

    def forward(self, x, skips):
        x_csc4 = self.csc4(x)
        d4_feat = self.HLFA4(x_csc4) 
        d4 = d4_feat + x     
        
        d3_up = self.LPCU3(d4)
        x3_gated = self.RGME3(g=d3_up, x=skips[0])
        d3_fused = d3_up + x3_gated

        d3_refined = self.sgu3(d3_fused)
        x3_csc = self.csc3(d3_refined)
        d3_feat = self.HLFA3(x3_csc) 
        d3 = d3_feat + d3_refined 
        
        d2_up = self.LPCU2(d3)
        x2_gated = self.RGME2(g=d2_up, x=skips[1])
        d2_fused = d2_up + x2_gated
        
        d2_refined = self.sgu2(d2_fused)
        x2_csc = self.csc2(d2_refined)
        d2_feat = self.HLFA2(x2_csc)
        d2 = d2_feat + d2_refined
        
        d1_up = self.LPCU1(d2)
        x1_gated = self.RGME1(g=d1_up, x=skips[2])
        d1_fused = d1_up + x1_gated
        
        d1_refined = self.sgu1(d1_fused)
        x1_csc = self.csc1(d1_refined)
        d1_feat = self.HLFA1(x1_csc)
        d1 = d1_feat + d1_refined

        return [d4, d3, d2, d1]


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, channels, opt, num_classes):
        super(Decoder, self).__init__()
        self.enhance = CNEC()
        self.conv1 = BasicConv2d(2 * 64, 64, 1)
        self.conv2 = BasicConv2d(2 * 128, 128, 1)
        self.conv3 = BasicConv2d(2 * 320, 320, 1)
        self.conv4 = BasicConv2d(2 * 512, 512, 1)

        self.conv_4 = BasicConv2d(channels[0], channels[0], 3, 1, 1)
        self.conv_3 = BasicConv2d(channels[1], channels[1], 3, 1, 1)
        self.conv_2 = BasicConv2d(channels[2], channels[2], 3, 1, 1)
        self.conv_1 = BasicConv2d(channels[3], channels[3], 3, 1, 1)

        self.head4 = nn.Conv2d(channels[0], num_classes, 2)
        self.head3 = nn.Conv2d(channels[1], num_classes, 2)
        self.head2 = nn.Conv2d(channels[2], num_classes, 2)
        self.head1 = nn.Conv2d(channels[3], num_classes, 2)

        self.LM = SGHiLo_Phi(channels=channels, opt=opt, lgag_ks=3)
        self.sig = nn.Sigmoid()

    def forward(self, pvta, pvtb, target_size):
        pvt_a1, pvt_a2, pvt_a3, pvt_a4 = self.enhance(pvta)
        pvt_b1, pvt_b2, pvt_b3, pvt_b4 = self.enhance(pvtb)

        layer_1 = self.conv_1(self.conv1(torch.cat((pvt_a1, pvt_b1), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((pvt_a2, pvt_b2), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((pvt_a3, pvt_b3), dim=1)))
        layer_4 = self.conv_4(self.conv4((torch.cat((pvt_a4, pvt_b4), dim=1))))
        outs = self.LM(layer_4, [layer_3, layer_2, layer_1])

        u4 = self.head4(outs[0])
        u3 = self.head3(outs[1])
        u2 = self.head2(outs[2])
        u1 = self.head1(outs[3])

        H_target, W_target = target_size

        u4 = F.interpolate(u4, size=(H_target, W_target), mode='bilinear', align_corners=False)
        u3 = F.interpolate(u3, size=(H_target, W_target), mode='bilinear', align_corners=False)
        u2 = F.interpolate(u2, size=(H_target, W_target), mode='bilinear', align_corners=False)
        u1 = F.interpolate(u1, size=(H_target, W_target), mode='bilinear', align_corners=False)

        return [self.sig(u4), self.sig(u3), self.sig(u2), self.sig(u1)]


class SAFENet_FD(nn.Module):
    def __init__(self, opt):
        super(SAFENet_FD, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(channels=[512, 320, 128, 64], num_classes=opt.num_classes, opt=opt)

    def forward(self, A_B):
        A = A_B[:, :3, :, :]
        B = A_B[:, 3:, :, :]
        H, W = A.shape[2], A.shape[3] 
        
        pvta, pvtb = self.encoder(A, B)
        pred = self.decoder(pvta, pvtb, (H, W))
        final_prediction = pred[3]
        return final_prediction, pvta, pvtb

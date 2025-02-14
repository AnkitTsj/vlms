import torch
import torch.nn as nn
import torch.nn.functional as F

from ex_model import EnhancedAttention3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InvertedResidual(nn.Module):  # Inverted Residual Block
    def __init__(self, inp, oup, stride, expand_ratio, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert stride == 1, "Stride in MBConv4D diagram seems to be always 1 based on residual connection."
        # print(f"the expand ratio is {expand_ratio}")
        hidden_dim = int(round(inp * expand_ratio))


        layers = []
        # pw - Expansion 1x1 Conv
        layers.extend([
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            norm_layer(hidden_dim),
            act_layer(),  # GeLU after expansion
        ])
        # pw-linear - Projection 1x1 Conv
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),  # BN after projection
        ])
        self.conv = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return x + self.conv(x)  # Residual connection addition




class Attention3D(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias).to(device)
        self.attn_drop = nn.Dropout(attn_drop).to(device)
        self.proj = nn.Linear(dim, dim).to(device)
        self.proj_drop = nn.Dropout(proj_drop).to(device)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratio,
                                     stride=sr_ratio, groups=dim).to(device)

            self.sr_norm = nn.LayerNorm(dim).to(device)
            self.sr_linear = nn.Linear(dim, head_dim).to(device)

    def forward(self, x):
        # x: (B, N, C) where N = H * W
        B, N, C = x.shape
        # Compute q, k, v from the full tokens.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # each is (B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.sr_ratio > 1:
            spatial_size = int(N ** 0.5)
            x_ = x.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
            # Apply convolution to reduce spatial resolution.
            x_ = self.sr_conv(x_)  # (B, C, H', W')
            # Flatten spatial dimensions: (B, C, N')
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)  # (B, N', C)
            # Project from C -> head_dim so that inner dimensions match.
            x_ = self.sr_linear(x_)  # (B, N', head_dim)

            # Expand x_ to have a head dimension: (B, 1, N', head_dim) -> (B, num_heads, N', head_dim)
            x_expanded = x_.unsqueeze(1).expand(B, self.num_heads, -1, -1)

            # For keys:
            # k is (B, num_heads, N, head_dim).
            scores_k = (x_expanded @ k.transpose(-2, -1))  # (B, num_heads, N', N)
            scores_k = scores_k.softmax(dim=-1)
            k = scores_k @ k  # (B, num_heads, N', head_dim)

            # Similarly for values:
            scores_v = (x_expanded @ v.transpose(-2, -1))  # (B, num_heads, N', N)
            scores_v = scores_v.softmax(dim=-1)
            v = scores_v @ v  # (B, num_heads, N', head_dim)

        # so attn becomes (B, num_heads, N, N').
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Multiply attention weights with v: (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MBConvBlock4D(nn.Module):
    def __init__(self, dim, drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d,
                 expand_ratio=4, pooling=nn.AvgPool2d(kernel_size=2,
                                                      stride=2)):  # Added pooling as argument for flexibility, default AvgPool2d
        super().__init__()
        self.pooling = pooling.to(device)  # Pooling layer before InvertedResidual
        self.norm1 = norm_layer(
            dim).to(device)
        self.conv = InvertedResidual(dim, dim, stride=1, expand_ratio=expand_ratio, act_layer=act_layer,
                                     norm_layer=norm_layer)  # Inverted Residual
        self.drop_path = nn.DropPath(drop_path).to(device) if drop_path > 0. else nn.Identity().to(device)

    def forward(self, x):
        pooled_x = self.pooling(x)  # Apply pooling first
        pooled_x = pooled_x.repeat(1, 1, 2, 2)
        # print(f"the shape of x is {x.shape} and pooled x is {pooled_x.shape}")

        # print(pooled_x.shape)
        x = self.conv(self.norm1(pooled_x + x))  # Apply BN then InvertedResidual to pooled input
        x = self.drop_path(x)
        return x  # Residual connection using pooled input


class MBConvBlock3D(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., norm_layer=nn.LayerNorm, sr_ratio=1,enhanced_attn = False):
        super().__init__()
        self.norm1 = norm_layer(dim).to(device)
        if enhanced_attn:
            self.attn = EnhancedAttention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                proj_drop=drop, sr_ratio=sr_ratio)
        else:
            self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim).to(device)
        self.actn_layer = nn.GELU()
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim,dim)
        self.linear_3 = nn.Linear(dim,dim)
        self.linear_4 = nn.Linear(dim,dim)

    def forward(self, x):
        prev_shape = x.shape
        x = x + self.linear_2(self.attn(self.linear_1(self.norm1(x.flatten(2).transpose(1, 2))))).reshape(x.shape)  # Attention in 3D
        x = self.norm2(x.permute(0,2,3,1))
        x = self.linear_3(x)
        x = self.actn_layer(x)
        x_ = self.linear_4(x).reshape(prev_shape)
        x = x.reshape(prev_shape) + x_  # MBConv residual connection
        return x




# --- EfficientFormer Model ---
class EfficientFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 5, 3, 3], dims=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],enhance_attn=False, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 sr_ratios=[8, 4, 2, 1], expand_ratios=[4, 4, 4, 4]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Patch embedding (Conv stem - two 3x3 convs with stride 2)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),  # Stride 2
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=2, padding=1),  # Stride 2, total stride 4
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        ).to(device)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # 4D Partition - CONV-net style (MBConvBlock4D blocks - DIAGRAM VERSION)
        self.stage1 = nn.ModuleList([
            MBConvBlock4D(dim=dims[0], mlp_ratio=mlp_ratios[0], drop=drop_rate, drop_path=dpr[i],
                          pooling=nn.AvgPool2d(kernel_size=2, stride=2))
            # Added pooling here, adjust parameters as needed
            for i in range(depths[0])]).to(device)

        # Downsample layer (Conv with stride 2) -
        self.downsample1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.GELU()
        ).to(device)

        # 4D Partition - CONV-net style
        self.stage2 = nn.ModuleList([
            MBConvBlock4D(dim=dims[1], mlp_ratio=mlp_ratios[1], drop=drop_rate, drop_path=dpr[sum(depths[:1]) + i],
                          pooling=nn.AvgPool2d(kernel_size=2, stride=2))
            # Added pooling here, adjust parameters as needed
            for i in range(depths[1])]).to(device)

        # Downsample layer (Conv with stride 2)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.GELU()
        ).to(device)

        # 3D Partition - Transformer style (MBConvBlock3D blocks with Attention)
        self.stage3 = nn.ModuleList([
            MBConvBlock3D(dim=dims[2], num_heads=num_heads[2], qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          attn_drop=attn_drop_rate, sr_ratio=sr_ratios[2],
                          drop=drop_rate,enhanced_attn=enhance_attn)
            for i in range(depths[2])]).to(device)

        # Downsample layer (Conv with stride 2) -
        self.downsample3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[3]),
            nn.GELU()
        ).to(device)

        # 3D Partition - Transformer style (MBConvBlock3D blocks with Attention)
        self.stage4 = nn.ModuleList([
            MBConvBlock3D(dim=dims[3], num_heads=num_heads[3], qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          attn_drop=attn_drop_rate, sr_ratio=sr_ratios[3],
                          drop=drop_rate,enhanced_attn=enhance_attn)
            for i in range(depths[3])]).to(device)

        self.norm = norm_layer(dims[-1]).to(device)  # Final Norm
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)  # Avg pool to get 1x1 feature map
        self.head = nn.Linear(dims[-1], num_classes).to(device) if num_classes > 0 else nn.Identity().to(device)  # Classification head

        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.patch_embed(x)  # Patch Embedding
        # print("patching done.")
        for blk in self.stage1:  # Stage 1 - MBConvBlock4D
            # print(blk)
            x = blk(x)
        x = self.downsample1(x)  # Downsample
        # print("x is sampled through stage 1")
        for blk in self.stage2:  # Stage 2 - MBConvBlock4D
            x = blk(x)
        x = self.downsample2(x)  # Downsample
        # print("x is sampled through stage 2")
        for blk in self.stage3:  # Stage 3 - MBConvBlock3D
            x = blk(x)
        # print("x is sampled through stage 3-half")
        x = self.downsample3(x)  # Downsample
        # print("x is sampled through stage 3")
        for blk in self.stage4:  # Stage 4 - MBConvBlock3D
            x = blk(x)

        x = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(x.shape)  # Final norm & reshape back to 4D
        x = self.avgpool(x)  # Avg Pool
        return x

    def forward(self, x):
        # print("yes")
        x = self.forward_features(x)
        x = self.head(x.flatten(1))  # Classification Head
        return x


# --- Helper function to create EfficientFormer model ---
def efficientformer_v1(**kwargs):  # Example config, adjust as needed
    model = EfficientFormer(depths=[3, 5, 3, 3], dims=[32, 64, 128, 256], num_heads=[1, 2, 4, 8],
                            mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1],enhance_attn=False)
    return model


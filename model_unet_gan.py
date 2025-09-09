import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


# ─────────────────────────── 基础模块 ─────────────────────────── #
class ResBlock(nn.Module):
    """Norm → ReLU → Conv3×3 → Norm → ReLU → Conv3×3 → 加残差 → ReLU"""

    def __init__(self, in_ch: int, out_ch: int, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.same = in_ch == out_ch
        self.norm1 = norm_layer(in_ch)
        self.norm2 = norm_layer(out_ch)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        self.match = nn.Identity() if self.same else nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor):
        residual = self.match(x)
        out = self.relu(self.norm1(x))
        out = self.conv1(out)
        out = self.relu(self.norm2(out))
        out = self.conv2(out)
        out = self.relu(out + residual)
        return out


class DownBlock(nn.Module):
    """ResBlock → MaxPool2d (移除了注意力机制)"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        feat = self.res(x)
        skip = feat  # 直接使用特征作为跳跃连接
        down = self.pool(feat)
        return down, skip


class UpBlock(nn.Module):
    """ConvT2d ↑2 → concat(skip) → ResBlock (移除了注意力机制)"""

    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # 上采样
        self.up = nn.ConvTranspose2d(prev_ch, skip_ch, 2, stride=2)
        # 特征融合后的处理
        self.res = ResBlock(skip_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        # 上采样
        x = self.up(x)

        # 调整尺寸
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bicubic", align_corners=False)

        # 直接拼接，不使用任何注意力机制
        x = torch.cat([skip, x], dim=1)

        return self.res(x)


# ============================ Basic UNet ============================ #
class UNet5DownWithGating(nn.Module):
    """基础UNet，移除了所有注意力机制"""

    def __init__(self, input_nc: int = 3, output_nc: int = 3):
        """
        Args:
            input_nc: 输入通道数
            output_nc: 输出通道数
        """
        super().__init__()

        ch = [32, 64, 128, 256, 512]

        # Encoder
        self.d1 = DownBlock(input_nc, ch[0])
        self.d2 = DownBlock(ch[0], ch[1])
        self.d3 = DownBlock(ch[1], ch[2])
        self.d4 = DownBlock(ch[2], ch[3])

        # Bottleneck - 移除了注意力机制
        self.bottleneck = ResBlock(ch[3], ch[4])

        # Decoder - 移除了注意力机制
        self.u1 = UpBlock(ch[4], ch[3], ch[3])
        self.u2 = UpBlock(ch[3], ch[2], ch[2])
        self.u3 = UpBlock(ch[2], ch[1], ch[1])
        self.u4 = UpBlock(ch[1], ch[0], ch[0])

        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch[0], output_nc, 1),
            # nn.Tanh()
            # nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor):
        # Encoder
        d1, s1 = self.d1(x)  # [B, 32, H/2, W/2], [B, 32, H, W]
        d2, s2 = self.d2(d1)  # [B, 64, H/4, W/4], [B, 64, H/2, W/2]
        d3, s3 = self.d3(d2)  # [B, 128, H/8, W/8], [B, 128, H/4, W/4]
        d4, s4 = self.d4(d3)  # [B, 256, H/16, W/16], [B, 256, H/8, W/8]

        # Bottleneck
        bott = self.bottleneck(d4)  # [B, 512, H/16, W/16]

        # Decoder
        u1 = self.u1(bott, s4)  # [B, 256, H/8, W/8]
        u2 = self.u2(u1, s3)  # [B, 128, H/4, W/4]
        u3 = self.u3(u2, s2)  # [B, 64, H/2, W/2]
        u4 = self.u4(u3, s1)  # [B, 32, H, W]

        # 主输出
        main_output = self.out_conv(u4)
        return main_output


# 兼容旧接口
class UnetGenerator(UNet5DownWithGating):
    def __init__(self, input_nc: int, output_nc: int, num_downs: int = 4,
                 ngf: int = 32, norm_layer=nn.InstanceNorm2d, use_dropout: bool = False,
                 **kwargs):
        """保持与原始接口的兼容性"""
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc
        )


# ──────────────────────── PatchGAN 判别器 ──────────────────────── #
class NLayerDiscriminator(nn.Module):
    """PatchGAN 鉴别器，首层 3→32 通道，深度 3 层 (kw=4, stride=2)。"""

    def __init__(self, input_nc: int = 3, ndf: int = 32, n_layers: int = 3,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        # 判断是否需要 bias
        use_bias = (
                           isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm2d
                   ) or norm_layer == nn.InstanceNorm2d

        kw, pad = 4, 1  # kernel_size=4, padding=1  (保证 Patch size = 70x70)

        layers = [
            # 第一层：3 → ndf (32)，不做归一化
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=pad),
            nn.LeakyReLU(0.2, True),
        ]

        # 中间层：通道逐层翻倍，stride=2
        nf_mult_prev, nf_mult = 1, 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=pad, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        # 最后一层：stride=1，保持尺寸
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=pad, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            # 输出 1 通道的 Patch map
            nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=pad),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容 5D 输入 (B,N,C,H,W)
        reshape = x.dim() == 5
        if reshape:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
        out = self.model(x)
        if reshape:
            out = out.view(b, n, 1, *out.shape[-2:])
        return out


# ──────────────────────── 测试和使用示例 ──────────────────────── #
def test_model():
    """测试模型功能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=== 测试基础UNet (移除注意力) ===")

    # 创建模型
    model = UNet5DownWithGating(
        input_nc=3,
        output_nc=3
    ).to(device)

    # 测试输入
    x = torch.randn(2, 3, 256, 256).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 测试判别器
    print("\n=== 测试判别器 ===")
    discriminator = NLayerDiscriminator(
        input_nc=3,
        ndf=32,
        n_layers=3
    ).to(device)

    with torch.no_grad():
        d_output = discriminator(output)
        print(f"判别器输出形状: {d_output.shape}")

    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"判别器参数量: {d_params:,}")


if __name__ == "__main__":
    test_model()
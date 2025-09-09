import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


# ───────────────────────── 注意力模块 ────────────────────────── #
class ChannelAttention(nn.Module):
    """通道注意力 (CBAM)，兼容 (B,C,H,W) 与 (B,N,C,H,W)"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshape = x.dim() == 5
        if reshape:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
        att = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        out = x * att
        if reshape:
            out = out.view(b, n, c, h, w)
        return out


class SpatialAttention(nn.Module):
    """空间注意力 (CBAM)"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * att


# ───────────────────────── Gating Signal模块 ────────────────────────── #
class GatingSignal(nn.Module):
    """生成门控信号，用于注意力机制"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionBlock(nn.Module):
    """注意力块，使用Gating Signal进行特征选择"""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()

        # 门控信号处理
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        # 跳跃连接处理
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        # 注意力权重生成
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: 来自解码器上一层的门控信号 (query)
            skip: 来自编码器的跳跃连接特征
        Returns:
            加权后的跳跃连接特征
        """
        # 处理门控信号
        g1 = self.W_gate(gate)

        # 处理跳跃连接
        x1 = self.W_skip(skip)

        # 确保尺寸匹配
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bicubic', align_corners=False)

        # 计算注意力权重
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # 应用注意力权重
        return skip * psi


class CrossAttentionGate(nn.Module):
    """交叉注意力门控模块，更复杂的特征融合"""

    def __init__(self, gate_channels: int, skip_channels: int, reduction: int = 8):
        super().__init__()

        inter_channels = max(gate_channels // reduction, skip_channels // reduction, 1)

        # Query from gate (decoder feature)
        self.query_conv = nn.Conv2d(gate_channels, inter_channels, 1)
        # Key from skip (encoder feature)
        self.key_conv = nn.Conv2d(skip_channels, inter_channels, 1)
        # Value from skip (encoder feature)
        self.value_conv = nn.Conv2d(skip_channels, skip_channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: 门控信号 [B, C_g, H_g, W_g]
            skip: 跳跃连接 [B, C_s, H_s, W_s]
        """
        B, C_s, H_s, W_s = skip.size()

        # 调整gate尺寸匹配skip
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=(H_s, W_s), mode='bicubic', align_corners=False)

        # 生成Q, K, V
        query = self.query_conv(gate).view(B, -1, H_s * W_s).permute(0, 2, 1)  # [B, N, C']
        key = self.key_conv(skip).view(B, -1, H_s * W_s)  # [B, C', N]
        value = self.value_conv(skip).view(B, -1, H_s * W_s)  # [B, C_s, N]

        # 计算注意力
        attention = torch.bmm(query, key)  # [B, N, N]
        attention = self.softmax(attention)

        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C_s, N]
        out = out.view(B, C_s, H_s, W_s)

        # 残差连接
        out = self.gamma * out + skip

        return out


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
    """ResBlock → CA+SA(skip) → MaxPool2d"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        feat = self.res(x)
        skip = self.sa(self.ca(feat))
        down = self.pool(feat)
        return down, skip


class UpBlock(nn.Module):
    """ConvT2d ↑2 → Gating Signal → Attention → concat(skip) → ResBlock"""

    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int,
                 use_attention: bool = True, attention_type: str = "simple"):
        super().__init__()

        self.use_attention = use_attention
        self.attention_type = attention_type

        # 上采样
        self.up = nn.ConvTranspose2d(prev_ch, skip_ch, 2, stride=2)

        # 门控信号生成
        if use_attention:
            self.gating = GatingSignal(skip_ch, skip_ch)

            if attention_type == "simple":
                # 简单注意力块
                self.attention = AttentionBlock(
                    gate_channels=skip_ch,
                    skip_channels=skip_ch,
                    inter_channels=skip_ch // 2
                )
            elif attention_type == "cross":
                # 交叉注意力门控
                self.attention = CrossAttentionGate(
                    gate_channels=skip_ch,
                    skip_channels=skip_ch
                )

        # 特征融合后的处理
        self.res = ResBlock(skip_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        # 上采样
        x = self.up(x)

        # 调整尺寸
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bicubic", align_corners=False)

        if self.use_attention:
            # 生成门控信号 (Query)
            gate_signal = self.gating(x)

            # 应用注意力机制
            attended_skip = self.attention(gate_signal, skip)

            # 拼接处理后的特征
            x = torch.cat([attended_skip, x], dim=1)
        else:
            # 直接拼接
            x = torch.cat([skip, x], dim=1)

        return self.res(x)


class MultiScaleUpBlock(nn.Module):
    """多尺度上采样块，结合多种注意力机制"""

    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int):
        super().__init__()

        # 多尺度上采样
        self.up_2x = nn.ConvTranspose2d(prev_ch, skip_ch, 2, stride=2)
        self.up_4x = nn.ConvTranspose2d(prev_ch, skip_ch, 4, stride=4)

        # 多尺度特征融合
        self.scale_fusion = nn.Conv2d(skip_ch * 2, skip_ch, 1)

        # 门控信号和注意力
        self.gating = GatingSignal(skip_ch, skip_ch)
        self.channel_att = ChannelAttention(skip_ch)
        self.spatial_att = SpatialAttention()
        self.cross_att = CrossAttentionGate(skip_ch, skip_ch)

        # 最终处理
        self.res = ResBlock(skip_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        H, W = skip.shape[2:]

        # 多尺度上采样
        up_2x = F.interpolate(self.up_2x(x), size=(H, W), mode='bicubic', align_corners=False)
        up_4x = F.interpolate(self.up_4x(x), size=(H, W), mode='bicubic', align_corners=False)

        # 融合多尺度特征
        multi_scale = self.scale_fusion(torch.cat([up_2x, up_4x], dim=1))

        # 门控信号
        gate = self.gating(multi_scale)

        # 多重注意力处理跳跃连接
        skip_enhanced = self.channel_att(skip)  # 通道注意力
        skip_enhanced = self.spatial_att(skip_enhanced)  # 空间注意力
        skip_enhanced = self.cross_att(gate, skip_enhanced)  # 交叉注意力

        # 最终融合
        fused = torch.cat([skip_enhanced, multi_scale], dim=1)
        return self.res(fused)


# ============================ Enhanced UNet ============================ #
class UNet5DownWithGating(nn.Module):
    """增强的UNet，每层上采样都使用Gating Signal"""

    def __init__(self, input_nc: int = 3, output_nc: int = 3,
                 attention_type: str = "simple", use_multi_scale: bool = False):
        """
        Args:
            input_nc: 输入通道数
            output_nc: 输出通道数
            attention_type: 注意力类型 ("simple", "cross")
            use_multi_scale: 是否使用多尺度上采样
        """
        super().__init__()

        ch = [32, 64, 128, 256, 512]

        # Encoder
        self.d1 = DownBlock(input_nc, ch[0])
        self.d2 = DownBlock(ch[0], ch[1])
        self.d3 = DownBlock(ch[1], ch[2])
        self.d4 = DownBlock(ch[2], ch[3])

        # Bottleneck with enhanced features
        self.bottleneck = nn.Sequential(
            ResBlock(ch[3], ch[4]),
            ChannelAttention(ch[4]),
            SpatialAttention()
        )

        # Decoder with Gating Signal
        if use_multi_scale:
            # 使用多尺度上采样块
            self.u1 = MultiScaleUpBlock(ch[4], ch[3], ch[3])
            self.u2 = MultiScaleUpBlock(ch[3], ch[2], ch[2])
            self.u3 = MultiScaleUpBlock(ch[2], ch[1], ch[1])
            self.u4 = MultiScaleUpBlock(ch[1], ch[0], ch[0])
        else:
            # 使用标准上采样块
            self.u1 = UpBlock(ch[4], ch[3], ch[3], True, attention_type)
            self.u2 = UpBlock(ch[3], ch[2], ch[2], True, attention_type)
            self.u3 = UpBlock(ch[2], ch[1], ch[1], True, attention_type)
            self.u4 = UpBlock(ch[1], ch[0], ch[0], True, attention_type)

        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch[0], output_nc, 1),
            # nn.Tanh()
        )

        # 深度监督分支 (可选)
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(ch[3], output_nc, 1),  # u1 output
            nn.Conv2d(ch[2], output_nc, 1),  # u2 output
            nn.Conv2d(ch[1], output_nc, 1),  # u3 output
        ])

    def forward(self, x: torch.Tensor, return_deep_features: bool = False):
        # Encoder
        d1, s1 = self.d1(x)  # [B, 32, H/2, W/2], [B, 32, H, W]
        d2, s2 = self.d2(d1)  # [B, 64, H/4, W/4], [B, 64, H/2, W/2]
        d3, s3 = self.d3(d2)  # [B, 128, H/8, W/8], [B, 128, H/4, W/4]
        d4, s4 = self.d4(d3)  # [B, 256, H/16, W/16], [B, 256, H/8, W/8]

        # Bottleneck
        bott = self.bottleneck(d4)  # [B, 512, H/16, W/16]

        # Decoder with Gating Signal
        u1 = self.u1(bott, s4)  # [B, 256, H/8, W/8]
        u2 = self.u2(u1, s3)  # [B, 128, H/4, W/4]
        u3 = self.u3(u2, s2)  # [B, 64, H/2, W/2]
        u4 = self.u4(u3, s1)  # [B, 32, H, W]

        # 主输出
        main_output = self.out_conv(u4)

        if return_deep_features:
            # 深度监督输出
            deep_outputs = []
            for i, (feat, conv) in enumerate(zip([u1, u2, u3], self.deep_supervision)):
                deep_out = F.interpolate(
                    torch.tanh(conv(feat)),
                    size=x.shape[2:],
                    mode='bicubic',
                    align_corners=False
                )
                deep_outputs.append(deep_out)

            return main_output, deep_outputs

        return main_output


# 兼容旧接口
class UnetGenerator(UNet5DownWithGating):
    def __init__(self, input_nc: int, output_nc: int, num_downs: int = 4,
                 ngf: int = 32, norm_layer=nn.InstanceNorm2d, use_dropout: bool = False,
                 **kwargs):
        """保持与原始接口的兼容性"""
        attention_type = kwargs.get('attention_type', 'simple')
        use_multi_scale = kwargs.get('use_multi_scale', False)
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            attention_type=attention_type,
            use_multi_scale=use_multi_scale
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

    # 测试不同配置
    configs = [
        {"attention_type": "simple", "use_multi_scale": False},
        {"attention_type": "cross", "use_multi_scale": False},
        {"attention_type": "simple", "use_multi_scale": True},
    ]

    for i, config in enumerate(configs):
        print(f"\n=== 测试配置 {i + 1}: {config} ===")

        # 创建模型
        model = UNet5DownWithGating(
            input_nc=3,
            output_nc=3,
            **config
        ).to(device)

        # 测试输入
        x = torch.randn(2, 3, 256, 256).to(device)

        # 前向传播
        with torch.no_grad():
            if config.get("use_multi_scale", False):
                output = model(x)
                print(f"输出形状: {output.shape}")
            else:
                output = model(x)
                deep_output, deep_features = model(x, return_deep_features=True)
                print(f"主输出形状: {output.shape}")
                print(f"深度特征数量: {len(deep_features)}")

        # 打印参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total_params:,}")


if __name__ == "__main__":
    test_model()
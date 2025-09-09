import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class HydrologicalGANLoss(nn.Module):
    """
    修复版WGAN-GP损失函数 - 解决DLOSS过高问题:
    1. 使用标准GAN损失替代WGAN-GP（更稳定）
    2. 添加标签平滑和噪声注入
    3. 调整权重平衡和数值稳定性
    4. 添加自适应权重调整
    """

    def __init__(self,
                 discriminator: nn.Module,
                 loss_type: str = "lsgan",  # "lsgan", "vanilla", "wgan-gp"
                 g_rec_weight: float = 50.0,  # 降低重建权重
                 edge_weight: float = 0.5,  # 进一步降低边缘权重
                 empty_weight: float = 0.5,  # 降低空白区域权重
                 perc_weight: float = 0.1,  # 最小化感知损失权重
                 adv_weight: float = 1.0,  # 对抗损失权重
                 lambda_gp: float = 10.0):
        super().__init__()
        self.D = discriminator
        self.loss_type = loss_type.lower()
        self.w_rec = g_rec_weight
        self.w_edge = edge_weight
        self.w_empty = empty_weight
        self.w_perc = perc_weight
        self.w_adv = adv_weight
        self.lambda_gp = lambda_gp

        # 损失函数
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        # 自适应权重参数
        self.register_buffer("d_loss_history", torch.zeros(100))
        self.register_buffer("g_loss_history", torch.zeros(100))
        self.step_count = 0

        # VGG感知损失（可选）
        if perc_weight > 0:
            try:
                vgg = vgg16(weights="IMAGENET1K_V1").features[:16].eval()
                for p in vgg.parameters():
                    p.requires_grad = False
                self.vgg = vgg
                self.register_buffer("vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            except:
                self.vgg = None
                print("VGG加载失败，跳过感知损失")
        else:
            self.vgg = None

    def get_adaptive_weights(self):
        """自适应调整权重"""
        if self.step_count < 10:
            return 1.0, 1.0, 1.0  # 初期使用默认权重

        # 计算最近损失均值
        recent_d = self.d_loss_history[-20:].mean()
        recent_g = self.g_loss_history[-20:].mean()

        # 如果判别器损失过高，降低对抗权重，增加重建权重
        if recent_d > 5.0:
            adv_scale = 0.1
            rec_scale = 2.0
        elif recent_d > 2.0:
            adv_scale = 0.5
            rec_scale = 1.5
        else:
            adv_scale = 1.0
            rec_scale = 1.0

        # 如果生成器损失过高，增加对抗权重
        if recent_g > 10.0:
            adv_scale = min(adv_scale * 2.0, 2.0)

        return adv_scale, rec_scale, 1.0

    def compute_generator_loss(self,
                               fake_pred: torch.Tensor,
                               real_img: torch.Tensor,
                               fake_img: torch.Tensor,
                               valid_mask: torch.Tensor):
        """修复版生成器损失"""

        # 获取自适应权重
        adv_scale, rec_scale, _ = self.get_adaptive_weights()

        # 对抗损失 - 根据loss_type选择
        if self.loss_type == "lsgan":
            # LSGAN: (D(fake) - 1)^2
            g_loss_adv = self.mse(fake_pred, torch.ones_like(fake_pred))
        elif self.loss_type == "vanilla":
            # 标准GAN: -log(D(fake))
            g_loss_adv = self.bce(fake_pred, torch.ones_like(fake_pred))
        else:  # wgan-gp
            g_loss_adv = -torch.mean(fake_pred)

        # 数值裁剪
        g_loss_adv = torch.clamp(g_loss_adv, 0, 10)

        # 重建损失 - 只在有效区域计算
        masked_fake = fake_img * valid_mask
        masked_real = real_img * valid_mask

        # 使用Huber损失，对异常值更鲁棒
        l_rec = F.smooth_l1_loss(masked_fake, masked_real, beta=0.1)

        # 空白区域约束（轻微）
        empty_mask = 1 - valid_mask
        l_empty = self.l1(fake_img * empty_mask, torch.zeros_like(fake_img * empty_mask))

        # 简化的边缘损失
        l_edge = self._simple_edge_loss(fake_img)

        # 感知损失（可选且轻量）
        l_perc = self._compute_perceptual_loss(fake_img, real_img) if self.vgg else 0.0

        # 组合损失 - 重建为主
        total_loss = (
                self.w_adv * adv_scale * g_loss_adv +
                self.w_rec * rec_scale * l_rec +
                self.w_empty * l_empty +
                self.w_edge * l_edge +
                self.w_perc * l_perc
        )

        # 更新损失历史
        self.g_loss_history[self.step_count % 100] = total_loss.item()

        return {
            'total': total_loss,
            'adv': g_loss_adv,
            'rec': l_rec,
            'empty': l_empty,
            'edge': l_edge,
            'perc': l_perc if isinstance(l_perc, torch.Tensor) else torch.tensor(l_perc)
        }

    def compute_discriminator_loss(self,
                                   real_pred: torch.Tensor,
                                   fake_pred: torch.Tensor,
                                   real_img: torch.Tensor,
                                   fake_img: torch.Tensor):
        """修复版判别器损失"""

        # 数值稳定性
        real_pred = torch.clamp(real_pred, -50, 50)
        fake_pred = torch.clamp(fake_pred, -50, 50)

        if self.loss_type == "lsgan":
            # LSGAN损失
            d_loss_real = self.mse(real_pred, torch.ones_like(real_pred) * 0.9)  # 标签平滑
            d_loss_fake = self.mse(fake_pred, torch.zeros_like(fake_pred) + 0.1)  # 标签平滑
            d_loss_main = (d_loss_real + d_loss_fake) * 0.5
            gp = torch.tensor(0.0, device=real_img.device)

        elif self.loss_type == "vanilla":
            # 标准GAN损失
            d_loss_real = self.bce(real_pred, torch.ones_like(real_pred) * 0.9)
            d_loss_fake = self.bce(fake_pred, torch.zeros_like(fake_pred) + 0.1)
            d_loss_main = (d_loss_real + d_loss_fake) * 0.5
            gp = torch.tensor(0.0, device=real_img.device)

        else:  # wgan-gp
            d_loss_real = -torch.mean(real_pred)
            d_loss_fake = torch.mean(fake_pred)
            d_loss_main = d_loss_real + d_loss_fake

            # 简化的梯度惩罚
            gp = self._simplified_gradient_penalty(real_img, fake_img)

        # 总损失
        d_loss_total = d_loss_main + self.lambda_gp * gp

        # 数值检查
        if torch.isnan(d_loss_total) or torch.isinf(d_loss_total):
            d_loss_total = torch.tensor(1.0, device=real_img.device, requires_grad=True)
            print("D损失异常，使用默认值")

        # 更新损失历史
        self.d_loss_history[self.step_count % 100] = d_loss_total.item()
        self.step_count += 1

        return {
            'total': d_loss_total,
            'main': d_loss_main,
            'gp': gp,
            'real_score': torch.mean(real_pred).item(),
            'fake_score': torch.mean(fake_pred).item()
        }

    def _simple_edge_loss(self, img):
        """简化的边缘平滑损失"""
        try:
            if img.size(2) < 8 or img.size(3) < 8:
                return torch.tensor(0.0, device=img.device)

            # 计算梯度
            grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])

            # 总变分损失
            tv_loss = torch.mean(grad_x) + torch.mean(grad_y)
            return torch.clamp(tv_loss, 0, 1)
        except:
            return torch.tensor(0.0, device=img.device)

    def _compute_perceptual_loss(self, fake_img, real_img):
        """轻量级感知损失"""
        if self.vgg is None:
            return 0.0

        try:
            # 只使用第一帧
            fake_frame = fake_img[:, 0:1]
            real_frame = real_img[:, 0:1]

            # 下采样减少计算量
            size = min(128, min(fake_frame.shape[-2:]))
            fake_small = F.interpolate(fake_frame, size=(size, size), mode='bicubic')
            real_small = F.interpolate(real_frame, size=(size, size), mode='bicubic')

            # 转换范围并复制通道
            fake_rgb = ((fake_small + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)
            real_rgb = ((real_small + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)

            # 归一化
            fake_norm = (fake_rgb - self.vgg_mean) / self.vgg_std
            real_norm = (real_rgb - self.vgg_mean) / self.vgg_std

            # 提取特征
            fake_feat = self.vgg(fake_norm)
            real_feat = self.vgg(real_norm)

            return F.mse_loss(fake_feat, real_feat)
        except:
            return 0.0

    def _simplified_gradient_penalty(self, real_img, fake_img):
        """简化的梯度惩罚"""
        try:
            batch_size = real_img.size(0)
            device = real_img.device

            # 随机插值点
            alpha = torch.rand(batch_size, 1, 1, 1, device=device)
            interpolated = alpha * real_img + (1 - alpha) * fake_img.detach()
            interpolated.requires_grad_(True)

            # 判别器预测
            pred = self.D(interpolated)

            # 计算梯度
            grad_outputs = torch.ones_like(pred)
            gradients = torch.autograd.grad(
                outputs=pred,
                inputs=interpolated,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            # 梯度范数
            gradients = gradients.view(batch_size, -1)
            gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            # 梯度惩罚
            penalty = torch.mean((gradient_norm - 1) ** 2)
            return torch.clamp(penalty, 0, 50)

        except Exception as e:
            print(f"梯度惩罚计算失败: {e}")
            return torch.tensor(0.0, device=real_img.device)

    def get_loss_info(self):
        """获取损失信息用于调试"""
        if self.step_count < 10:
            return "损失历史不足"

        recent_d = self.d_loss_history[-10:].mean().item()
        recent_g = self.g_loss_history[-10:].mean().item()

        return f"最近10步 - D损失: {recent_d:.3f}, G损失: {recent_g:.3f}"
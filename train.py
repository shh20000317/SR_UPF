import os
import logging
import functools
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from model_unet_gan import UnetGenerator, NLayerDiscriminator
from data_preprocessing import HydrologyDataProcessor, NUM_FRAMES
from loss_function import HydrologicalGANLoss

# ───────────────────────────────── PATHS & HYPER-PARAMETERS ──────────────────────────────────
PATHS: Dict[str, str] = {
    "train_hr": r"D:\Program Files\PyCharm\MODEL\DL\pythonProject\SR_UPF\10m",
    "train_lr": r"D:\Program Files\PyCharm\MODEL\DL\pythonProject\SR_UPF\100m",
    "save_dir": r"D:\Program Files\PyCharm\MODEL\DL\pythonProject\SR_UPF\checkpoints",
    "log_dir": r"D:\Program Files\PyCharm\MODEL\DL\pythonProject\SR_UPF\logs",
}

TRAIN: Dict[str, float] = {
    "batch": 2,  # 减小batch size
    "epochs": 300,
    "g_lr": 1e-3,  # 降低生成器学习率
    "d_lr": 1e-3,  # 大幅降低判别器学习率
    "betas": (0.5, 0.999),
    "d_steps": 1,  # 判别器更新次数
    "g_steps": 2,  # 增加生成器更新次数
    "warmup_epochs": 10,  # 预热轮次
}

MODEL: Dict[str, int] = {
    "in": NUM_FRAMES,
    "out": NUM_FRAMES,
    "base": 32,
}


class HydroTrainer:
    def __init__(self) -> None:
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.dev}")

        # 确保目录存在
        for d in (PATHS["save_dir"], PATHS["log_dir"], PATHS["out_dir"]):
            os.makedirs(d, exist_ok=True)

        self._init_logger()
        self._init_data()
        self._init_model()

        self.history: Dict[str, List[float]] = {
            "psnr": [], "ssim": [], "mse": [],
            "g_loss": [], "d_loss": [],
            "d_real_score": [], "d_fake_score": []
        }
        self.best_mse: float = float("inf")

    def _init_logger(self) -> None:
        log_file = os.path.join(PATHS["log_dir"], "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        )
        self.log = logging.getLogger("HydroGAN")

    def _init_data(self) -> None:
        full_ds = HydrologyDataProcessor(
            hr_root=PATHS["train_hr"],
            lr_root=PATHS["train_lr"],
            paths=PATHS
        )

        val_len = int(0.2 * len(full_ds))
        train_len = len(full_ds) - val_len
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=generator)

        self.train_ld = DataLoader(
            train_ds,
            batch_size=TRAIN["batch"],
            shuffle=True,
            num_workers=0,
            collate_fn=full_ds.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        self.val_ld = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=full_ds.collate_fn,
            pin_memory=True,
        )

        self.log.info(f"Dataset | train {len(train_ds)} | val {len(val_ds)}")

    def _init_model(self) -> None:
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)

        self.G = UnetGenerator(
            input_nc=MODEL["in"],
            output_nc=MODEL["out"],
            num_downs=4,
            ngf=MODEL["base"],
            norm_layer=norm_layer,
            use_dropout=False,
        ).to(self.dev)

        self.D = NLayerDiscriminator(
            input_nc=MODEL["out"],
            ndf=MODEL["base"],
            n_layers=3,
            norm_layer=norm_layer,
        ).to(self.dev)

        # 使用LSGAN损失（更稳定）
        self.crit = HydrologicalGANLoss(
            discriminator=self.D,
            loss_type="lsgan",  # 使用LSGAN而不是WGAN-GP
            g_rec_weight=50.0,  # 降低重建权重
            edge_weight=0.5,  # 最小化边缘损失
            empty_weight=0.5,  # 最小化空白损失
            perc_weight=0.1,  # 最小化感知损失
            adv_weight=1.0,  # 降低对抗损失权重
        ).to(self.dev)

        # 优化器 - 使用不同学习率
        self.optG = optim.Adam(self.G.parameters(), lr=TRAIN["g_lr"], betas=TRAIN["betas"])
        self.optD = optim.Adam(self.D.parameters(), lr=TRAIN["d_lr"], betas=TRAIN["betas"])

        # AMP scaler
        self.scalerG, self.scalerD = GradScaler(), GradScaler()

        # 学习率调度器 - 更保守
        self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(
            self.optG, mode='min', factor=0.8, patience=15, verbose=True
        )
        self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(
            self.optD, mode='min', factor=0.8, patience=15, verbose=True
        )

        self.log.info("Model & optimizers initialised with LSGAN loss")

    def _get_noise_factor(self, epoch):
        """获取噪声因子，早期训练时添加噪声提高稳定性"""
        if epoch <= TRAIN["warmup_epochs"]:
            return 0.1 * (1 - epoch / TRAIN["warmup_epochs"])
        return 0.0

    def _add_noise(self, img, factor):
        """添加少量噪声"""
        if factor > 0:
            noise = torch.randn_like(img) * factor
            return img + noise
        return img

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        self.G.train()
        self.D.train()

        g_loss_total, d_loss_total = 0.0, 0.0
        d_real_scores, d_fake_scores = [], []
        batch_count = 0

        noise_factor = self._get_noise_factor(epoch)

        for batch_idx, batch in enumerate(tqdm(self.train_ld, desc=f"Train Ep{epoch:03d}")):
            if batch is None:
                continue

            lr = batch["input"].to(self.dev)
            hr = batch["target"].to(self.dev)
            mask = batch["valid_mask"].to(self.dev)

            # 添加噪声（早期训练时）
            if noise_factor > 0:
                hr = self._add_noise(hr, noise_factor)

            # ===== 更新判别器 ===== #
            d_loss_batch = 0.0
            for _ in range(TRAIN["d_steps"]):
                try:
                    self.optD.zero_grad(set_to_none=True)

                    with autocast():
                        # 生成假图像（不需要梯度）
                        with torch.no_grad():
                            fake_img = self.G(lr).detach()

                        # 判别器预测
                        real_pred = self.D(hr)
                        fake_pred = self.D(fake_img)

                        # 计算判别器损失
                        d_loss_dict = self.crit.compute_discriminator_loss(
                            real_pred=real_pred,
                            fake_pred=fake_pred,
                            real_img=hr,
                            fake_img=fake_img,
                        )

                    d_loss = d_loss_dict['total']

                    # 检查损失有效性
                    if torch.isnan(d_loss) or torch.isinf(d_loss):
                        self.log.warning(f"Invalid D loss at batch {batch_idx}, skipping")
                        continue

                    # 反向传播
                    self.scalerD.scale(d_loss).backward()
                    self.scalerD.unscale_(self.optD)
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=0.5)  # 更严格的梯度裁剪
                    self.scalerD.step(self.optD)
                    self.scalerD.update()

                    d_loss_batch += d_loss.item()
                    d_real_scores.append(d_loss_dict['real_score'])
                    d_fake_scores.append(d_loss_dict['fake_score'])

                except Exception as e:
                    self.log.warning(f"D training error: {e}")
                    continue

            # ===== 更新生成器 ===== #
            g_loss_batch = 0.0
            for _ in range(TRAIN["g_steps"]):
                try:
                    self.optG.zero_grad(set_to_none=True)

                    with autocast():
                        fake_img = self.G(lr)
                        fake_pred = self.D(fake_img)

                        g_loss_dict = self.crit.compute_generator_loss(
                            fake_pred=fake_pred,
                            real_img=hr,
                            fake_img=fake_img,
                            valid_mask=mask,
                        )

                    g_loss = g_loss_dict['total']

                    if torch.isnan(g_loss) or torch.isinf(g_loss):
                        self.log.warning(f"Invalid G loss at batch {batch_idx}, skipping")
                        continue

                    self.scalerG.scale(g_loss).backward()
                    self.scalerG.unscale_(self.optG)
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
                    self.scalerG.step(self.optG)
                    self.scalerG.update()

                    g_loss_batch += g_loss.item()

                except Exception as e:
                    self.log.warning(f"G training error: {e}")
                    continue

            # 记录有效batch
            if g_loss_batch > 0 and d_loss_batch > 0:
                g_loss_total += g_loss_batch / TRAIN["g_steps"]
                d_loss_total += d_loss_batch / TRAIN["d_steps"]
                batch_count += 1

                # 详细日志
                if batch_idx % 20 == 0 and batch_idx > 0:
                    recent_real = np.mean(d_real_scores[-5:]) if len(d_real_scores) >= 5 else 0
                    recent_fake = np.mean(d_fake_scores[-5:]) if len(d_fake_scores) >= 5 else 0
                    self.log.info(f"Batch {batch_idx:03d} | "
                                  f"G: {g_loss_batch / TRAIN['g_steps']:.4f} | "
                                  f"D: {d_loss_batch / TRAIN['d_steps']:.4f} | "
                                  f"Real: {recent_real:.3f} | Fake: {recent_fake:.3f}")

                    # 输出损失组件信息
                    if hasattr(self.crit, 'get_loss_info'):
                        self.log.info(f"损失状态: {self.crit.get_loss_info()}")

        if batch_count == 0:
            self.log.error("No successful batches in this epoch!")
            return float('inf'), float('inf'), 0.0, 0.0

        # 计算平均值
        g_avg = g_loss_total / batch_count
        d_avg = d_loss_total / batch_count
        real_avg = np.mean(d_real_scores) if d_real_scores else 0.0
        fake_avg = np.mean(d_fake_scores) if d_fake_scores else 0.0

        # 更新学习率
        self.schedulerG.step(g_avg)
        self.schedulerD.step(d_avg)

        return g_avg, d_avg, real_avg, fake_avg

    # ───────────────────────────── validate ────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, float, float]:
        self.G.eval()

        psnr_list: List[float] = []
        ssim_list: List[float] = []
        mse_list: List[float] = []

        for batch in tqdm(self.val_ld, desc=f"Val   Ep{epoch:03d}"):
            if batch is None:
                continue

            try:
                lr = batch["input"].to(self.dev)
                hr = batch["target"].to(self.dev)
                mask = batch["valid_mask"].to(self.dev)

                with autocast():
                    pred = self.G(lr)

                # 转换为numpy并限制范围
                pred_np = pred.clamp(-1, 1).cpu().numpy()
                hr_np = hr.cpu().numpy()
                mask_np = mask.cpu().numpy()

                # 统一处理为4维: (B*N, C, H, W)
                if pred_np.ndim == 5:  # (B, N, C, H, W) -> (B*N, C, H, W)
                    B, N, C, H, W = pred_np.shape
                    pred_np = pred_np.reshape(-1, C, H, W)
                    hr_np = hr_np.reshape(-1, C, H, W)
                    # 扩展mask: (B, 1, H, W) -> (B*N, 1, H, W)
                    mask_np = np.repeat(mask_np, N, axis=0)

                # 现在所有数据都是 (B, C, H, W) 格式
                B, C, H, W = pred_np.shape

                # 逐样本计算指标
                for b in range(B):
                    # 获取当前样本
                    pred_sample = pred_np[b]  # (C, H, W)
                    hr_sample = hr_np[b]  # (C, H, W)
                    mask_sample = mask_np[b, 0]  # (H, W) - 取第一个通道作为mask

                    # 应用mask到所有通道
                    valid_pixels = mask_sample == 1

                    if np.sum(valid_pixels) == 0:
                        continue

                    # 逐通道计算指标然后平均
                    channel_psnr, channel_ssim, channel_mse = [], [], []

                    for c in range(C):
                        pred_valid = pred_sample[c][valid_pixels]
                        hr_valid = hr_sample[c][valid_pixels]

                        if len(pred_valid) > 0:
                            # 计算指标
                            try:
                                psnr_val = psnr(hr_valid, pred_valid, data_range=2.0)
                                # 对于1D数据，SSIM需要特殊处理
                                if len(pred_valid) > 1:
                                    ssim_val = ssim(hr_valid, pred_valid, data_range=2.0)
                                else:
                                    ssim_val = 1.0 if np.allclose(hr_valid, pred_valid) else 0.0
                                mse_val = mse(hr_valid, pred_valid)

                                # 检查数值有效性
                                if np.isfinite(psnr_val) and np.isfinite(ssim_val) and np.isfinite(mse_val):
                                    channel_psnr.append(psnr_val)
                                    channel_ssim.append(ssim_val)
                                    channel_mse.append(mse_val)
                            except Exception as e:
                                self.log.warning(f"Metric calculation error: {e}")
                                continue

                    if channel_psnr:  # 如果有有效计算
                        psnr_list.append(np.mean(channel_psnr))
                        ssim_list.append(np.mean(channel_ssim))
                        mse_list.append(np.mean(channel_mse))

            except Exception as e:
                self.log.warning(f"Validation batch error: {e}")
                continue

        if not psnr_list:
            self.log.warning(f"[Val] Ep{epoch:03d} | No valid samples for evaluation")
            return float('inf'), 0.0, 0.0

        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)
        mean_mse = np.mean(mse_list)

        self.log.info(f"[Val] Ep{epoch:03d} | PSNR {mean_psnr:.2f} | SSIM {mean_ssim:.4f} | MSE {mean_mse:.5f}")
        return mean_mse, mean_psnr, mean_ssim

    # ─────────────────────────── save checkpoint ────────────────────────────────────────────
    def _save_ckpt(self, epoch: int, is_best: bool = False) -> None:
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.G.state_dict(),
            "discriminator_state_dict": self.D.state_dict(),
            "optimizer_G_state_dict": self.optG.state_dict(),
            "optimizer_D_state_dict": self.optD.state_dict(),
            "scheduler_G_state_dict": self.schedulerG.state_dict(),
            "scheduler_D_state_dict": self.schedulerD.state_dict(),
            "scaler_G_state_dict": self.scalerG.state_dict(),
            "scaler_D_state_dict": self.scalerD.state_dict(),
            "history": self.history,
            "best_mse": self.best_mse,
        }

        # 保存当前检查点
        ckpt_path = os.path.join(PATHS["save_dir"], f"checkpoint_epoch_{epoch:03d}.pth")
        torch.save(checkpoint, ckpt_path)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(PATHS["save_dir"], "best_model.pth")
            torch.save(checkpoint, best_path)
            self.log.info(f"Best model saved at epoch {epoch} with MSE: {self.best_mse:.5f}")

        self.log.info(f"Checkpoint saved: {ckpt_path}")

    # ─────────────────────────────── main training loop ──────────────────────────────────────
    def run(self) -> None:
        """主训练循环"""
        self.log.info("Starting training...")

        for ep in range(1, TRAIN["epochs"] + 1):
            try:
                # 训练一个epoch
                g_loss, d_loss, real_avg, fake_avg = self._train_epoch(ep)
                # 或者 g_loss, d_loss, *_ = self._train_epoch(ep)

                # 验证
                val_mse, val_psnr, val_ssim = self._validate(ep)

                # 记录损失历史
                self.history["g_loss"].append(g_loss)
                self.history["d_loss"].append(d_loss)
                self.history["mse"].append(val_mse)
                self.history["psnr"].append(val_psnr)
                self.history["ssim"].append(val_ssim)

                # 记录日志
                self.log.info(f"[Epoch {ep:03d}] G_Loss: {g_loss:.4f} | D_Loss: {d_loss:.4f} | "
                              f"Val_MSE: {val_mse:.5f} | Val_PSNR: {val_psnr:.2f} | Val_SSIM: {val_ssim:.4f}")

                # 检查是否为最佳模型
                is_best = False
                if val_mse < self.best_mse:
                    self.best_mse = val_mse
                    is_best = True

                # 每10个epoch或最佳模型时保存检查点
                if ep % 10 == 0 or is_best:
                    self._save_ckpt(ep, is_best)

                # 早期停止检查（可选）
                if len(self.history["mse"]) > 20:
                    recent_mse = self.history["mse"][-10:]
                    if all(mse_val >= self.best_mse * 1.1 for mse_val in recent_mse):
                        self.log.info(f"Early stopping at epoch {ep} - no improvement in last 10 epochs")
                        break

            except Exception as e:
                self.log.error(f"Error in epoch {ep}: {e}")
                continue

        # 训练完成，保存最终检查点
        self._save_ckpt(TRAIN["epochs"], False)
        self.log.info("Training completed!")

        # 打印最佳结果
        if self.history["mse"]:
            best_epoch = np.argmin(self.history["mse"]) + 1
            self.log.info(f"Best results at epoch {best_epoch}: "
                          f"MSE={self.best_mse:.5f}, "
                          f"PSNR={self.history['psnr'][best_epoch - 1]:.2f}, "
                          f"SSIM={self.history['ssim'][best_epoch - 1]:.4f}")


# ───────────────────────────────── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        trainer = HydroTrainer()
        trainer.run()  # 启动训练
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
import os, logging
from typing import Dict, Tuple
from functools import lru_cache

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------- 尺寸常量 ---------- #
ORIG_H, ORIG_W = 1207, 1563     # 原 HR
PAD            = 2048           # 统一填充
NUM_FRAMES     = 3        # 帧数 t,t+1,t+2
# ----------------------------- #

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.FileHandler("data_process.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)


class HydrologyDataProcessor(Dataset):
    """
    · HR / LR 为单波段 .tif
    · LR 上采样→HR 尺寸→归一化到 [-1,1]→PAD 到 2048²
    · 返回 dict:
        input       (C,2048,2048)
        target      (C,2048,2048)
        valid_mask  (1,2048,2048)  1=有效像素
        norm_param  (2,) tensor [vmin,vmax]
    """

    # -------------------------------------------------------- #
    def __init__(self,
                 hr_root: str, lr_root: str,
                 phase: str = "train",
                 paths: Dict | None = None):
        if not paths:
            raise ValueError("paths 字典不能为空")
        self.paths = paths
        self.phase = phase.lower()

        self.hr_dir, self.lr_dir = self._phase_dirs(hr_root, lr_root)
        self.hr_files = sorted([f for f in os.listdir(self.hr_dir) if f.endswith(".tif")])
        self.lr_files = sorted([f for f in os.listdir(self.lr_dir) if f.endswith(".tif")])
        assert len(self.hr_files) == len(self.lr_files), "HR/LR 数目不一致"

        self.hr_paths = [os.path.join(self.hr_dir, f) for f in self.hr_files]
        self.lr_paths = [os.path.join(self.lr_dir, f) for f in self.lr_files]

        # 计算归一化区间
        self.vmin, self.vmax = self._scan_minmax()
        logger.info(f"自动归一化区间: [{self.vmin:.3f}, {self.vmax:.3f}]")

    # ---------------- dir helper ---------------- #
    def _phase_dirs(self, hr_root, lr_root):
        mp = {"train": ("train_hr", "train_lr"),
              "val":   ("val_hr",   "val_lr"),
              "test":  ("val_hr",   "val_lr")}
        hr_d = self.paths.get(mp[self.phase][0], hr_root)
        lr_d = self.paths.get(mp[self.phase][1], lr_root)
        if not (os.path.isdir(hr_d) and os.path.isdir(lr_d)):
            raise FileNotFoundError("阶段路径不存在")
        return hr_d, lr_d

    # ---------------- IO cache ---------------- #
    @staticmethod
    @lru_cache(maxsize=512)
    def _read(path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)

    # ---------------- min/max scan ---------------- #
    def _scan_minmax(self):
        vmin, vmax = np.inf, -np.inf
        for p in self.hr_paths:
            arr = self._read(p)
            vmin = min(vmin, arr.min())
            vmax = max(vmax, arr.max())
        return float(vmin), float(vmax)

    # ---------------- basic funcs ---------------- #
    def _norm(self, x):
        return (x - self.vmin) / (self.vmax - self.vmin + 1e-8) * 2 - 1

    def _pad(self, img: np.ndarray):
        H, W = img.shape
        canvas = np.zeros((PAD, PAD), img.dtype)
        canvas[:H, :W] = img
        return canvas

    # ---------------- Dataset ---------------- #
    def __len__(self):
        return len(self.hr_files) - (NUM_FRAMES  - 1)

    def __getitem__(self, idx):
        try:
            hr_ls, lr_ls = [], []
            for t in range(NUM_FRAMES):
                i = idx + t
                hr = self._read(self.hr_paths[i])
                lr = np.maximum(self._read(self.lr_paths[i]), 0)

                lr_up = F.interpolate(torch.from_numpy(lr)[None, None].float(),
                                      size=(ORIG_H, ORIG_W),
                                      mode="bicubic",
                                      align_corners=False).squeeze().numpy()

                hr_ls.append(self._pad(self._norm(hr)))
                lr_ls.append(self._pad(self._norm(lr_up)))

            hr_stk = np.stack(hr_ls, axis=0)  # (3,2048,2048)
            lr_stk = np.stack(lr_ls, axis=0)

            valid = np.zeros((1, PAD, PAD), np.float32)
            valid[:, :ORIG_H, :ORIG_W] = 1.  # 创建有效区域

            # 添加 HR 完整路径
            hr_full_path = self.hr_paths[idx]

            return {
                "input": torch.from_numpy(lr_stk).float(),
                "target": torch.from_numpy(hr_stk).float(),
                "valid_mask": torch.from_numpy(valid).float(),
                "norm_param": torch.tensor([self.vmin, self.vmax], dtype=torch.float32),
                "hr_file": hr_full_path
            }
        except Exception as e:
            logger.error(f"样本 {idx} 处理失败: {e}")
            return None

    # ---------------- collate ---------------- #
    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b]
        if not batch:
            return None
        collated_batch = {
            k: torch.stack([item[k] for item in batch]) if k != 'hr_file' else [item['hr_file'] for item in batch]
            for k in batch[0].keys()
        }
        return collated_batch


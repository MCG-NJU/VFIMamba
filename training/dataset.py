"""
Custom triplet dataset for VFIMamba fine-tuning.

ASSUMED LAYOUT (matches the Vimeo90k-triplet convention this codebase
family - EMA-VFI / VFIMamba - is built around):

    data_root/
        seq_0001/
            im1.png   # frame t-1
            im2.png   # frame t   (ground truth, the interpolated frame)
            im3.png   # frame t+1
        seq_0002/
            ...

Every subfolder of data_root that contains im1.png/im2.png/im3.png is
treated as one training triplet. If your 2k images aren't organized this
way (flat files, different names, need extraction from video first),
tell me the actual layout and I'll adjust this loader instead of guessing
further.

The 80/20 split is deterministic (seeded shuffle) so re-running train.py
always reproduces the same train/val partition.
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VFIDataset(Dataset):
    def __init__(self, data_root, mode='train', crop_size=256, val_split=0.2, seed=42):
        assert mode in ('train', 'val')
        assert crop_size % 32 == 0, 'crop_size must be a multiple of 32 for this architecture'
        self.data_root = data_root
        self.mode = mode
        self.crop_size = crop_size

        all_seqs = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
            and all(os.path.exists(os.path.join(data_root, d, f'im{i}.png')) for i in (1, 2, 3))
        ])
        if not all_seqs:
            raise RuntimeError(
                f'No triplet folders found under {data_root}. '
                'Expected each subfolder to contain im1.png, im2.png, im3.png.'
            )

        rng = random.Random(seed)
        shuffled = all_seqs[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_split)))
        val_seqs = set(shuffled[:n_val])

        self.seqs = [s for s in all_seqs if (s in val_seqs) == (mode == 'val')]

    def __len__(self):
        return len(self.seqs)

    @staticmethod
    def _load(path):
        img = cv2.imread(path)  # BGR HxWx3, kept as-is to match the pretrained model's channel convention
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _augment(self, img0, gt, img1):
        h, w, _ = gt.shape
        ch = cw = self.crop_size
        if h < ch or w < cw:
            raise ValueError(f'crop_size {self.crop_size} larger than source image {h}x{w}')

        if self.mode == 'train':
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
        else:
            top = (h - ch) // 2
            left = (w - cw) // 2

        def crop(im):
            return im[top:top + ch, left:left + cw]

        img0, gt, img1 = crop(img0), crop(gt), crop(img1)

        if self.mode == 'train':
            if random.random() < 0.5:  # horizontal flip
                img0, gt, img1 = img0[:, ::-1], gt[:, ::-1], img1[:, ::-1]
            if random.random() < 0.5:  # vertical flip
                img0, gt, img1 = img0[::-1], gt[::-1], img1[::-1]
            if random.random() < 0.5:  # reverse temporal order
                img0, img1 = img1, img0

        return img0.copy(), gt.copy(), img1.copy()

    def __getitem__(self, idx):
        seq_dir = os.path.join(self.data_root, self.seqs[idx])
        img0 = self._load(os.path.join(seq_dir, 'im1.png'))
        gt = self._load(os.path.join(seq_dir, 'im2.png'))
        img1 = self._load(os.path.join(seq_dir, 'im3.png'))

        img0, gt, img1 = self._augment(img0, gt, img1)

        def to_tensor(im):
            return torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255.0)

        return to_tensor(img0), to_tensor(gt), to_tensor(img1)
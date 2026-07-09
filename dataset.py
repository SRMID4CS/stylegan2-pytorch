from io import BytesIO
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class NpyMelDataset(Dataset):
    """float32 mel-canvas .npy files written by prepare_audio_data.py.

    Samples are already normalized to [-1,1] and single-channel — no transform of
    any kind is applied (no flip, no ToTensor rescale, no Normalize).
    """

    def __init__(self, path, expected_shape=None):
        self.files = sorted(Path(path).glob("*.npy"))
        if not self.files:
            raise IOError(f"[AUDIO] no .npy files found in {path}")
        self.expected_shape = tuple(expected_shape) if expected_shape is not None else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        arr = np.load(self.files[index])
        if arr.dtype != np.float32:
            raise ValueError(f"[AUDIO] {self.files[index]}: dtype {arr.dtype}, expected float32")
        if self.expected_shape is not None and tuple(arr.shape) != self.expected_shape:
            raise ValueError(
                f"[AUDIO] {self.files[index]}: shape {tuple(arr.shape)}, expected {self.expected_shape}"
            )
        return torch.from_numpy(arr)

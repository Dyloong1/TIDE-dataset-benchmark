"""
JHU Rollout Dataset for 1-to-1 prediction with continuous temporal split.

Splits 200 timestamps (5.02..7.01 at 0.01 step) into 139 train / 30 val / 31 test
sequentially (no chunking). Each __getitem__ returns a (t, t+1) 1-frame pair,
shape (1, C, Z, H, W) for both input and output.

Channel selection is configurable: pass indicators=['u','v','p'] to drop w.

Loads .npy volumes directly from `data_dir` and normalizes per-channel.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


NORM_STATS = {
    'u': {'mean': 0.000864, 'std': 0.6385},
    'v': {'mean': 0.00121,  'std': 0.7478},
    'w': {'mean': 0.00211,  'std': 0.6766},
    'p': {'mean': 0.000372, 'std': 0.4118},
}


def build_timestamps(n=200, start=5.02, step=0.01):
    return [round(start + i * step, 3) for i in range(n)]


def split_timestamps(all_ts, n_train=139, n_val=30, n_test=31):
    assert len(all_ts) == n_train + n_val + n_test, \
        f'{len(all_ts)} != {n_train}+{n_val}+{n_test}'
    return {
        'train': all_ts[:n_train],
        'val': all_ts[n_train:n_train + n_val],
        'test': all_ts[n_train + n_val:],
    }


class JHURolloutDataset(Dataset):
    """1→1 dataset: each item is a (frame_t, frame_{t+1}) pair."""

    def __init__(self, data_dir, split='train',
                 indicators=('u', 'v', 'w', 'p'),
                 n_train=139, n_val=30, n_test=31):
        self.data_dir = Path(data_dir)
        self.indicators = list(indicators)

        all_ts = build_timestamps(n=n_train + n_val + n_test)
        splits = split_timestamps(all_ts, n_train, n_val, n_test)
        if split not in splits:
            raise ValueError(f'Unknown split {split!r}')
        self.timestamps = splits[split]
        self.split = split

    def __len__(self):
        return len(self.timestamps) - 1  # number of consecutive pairs

    def _load_frame(self, ts):
        frames = []
        for ind in self.indicators:
            path = self.data_dir / f"{ind}_{ts:.3f}.npy"
            data = np.load(str(path)).astype(np.float32)
            stats = NORM_STATS[ind]
            data = (data - stats['mean']) / (stats['std'] + 1e-8)
            frames.append(data)
        return np.stack(frames, axis=0)  # (C, Z, H, W)

    def __getitem__(self, idx):
        ts_in = self.timestamps[idx]
        ts_out = self.timestamps[idx + 1]
        input_frame = self._load_frame(ts_in)    # (C, Z, H, W)
        output_frame = self._load_frame(ts_out)

        # Add T dim so shape is (T=1, C, Z, H, W)
        input_dense = torch.from_numpy(input_frame).unsqueeze(0)
        output_dense = torch.from_numpy(output_frame).unsqueeze(0)

        return {
            'input_dense': input_dense,
            'output_dense': output_dense,
            'ts_in': ts_in,
            'ts_out': ts_out,
        }

    def load_sequence(self):
        """Load the entire split as a single tensor for rollout evaluation.

        Returns: (T, C, Z, H, W) tensor of normalized frames.
        """
        frames = [self._load_frame(ts) for ts in self.timestamps]
        return torch.from_numpy(np.stack(frames, axis=0))

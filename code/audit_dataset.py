from audit_utils import audit_file_loading, audit_dataset_wrapper
audit_dataset_wrapper(__file__)
# PLEASE DO NOT EDIT THE FIRST THREE LINES
# The following code may be edited, but please ensure the function 
# audit_utils.audit_file_loading is called before loading the file 
# each time.

from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path


def npz_to_tensor(img):
    """NPZ image [0,1] → tensor (1,H,W) [−1,1]."""
    t = torch.from_numpy(img).float().unsqueeze(0)  # (1, H, W)
    return t * 2.0 - 1.0 


class UnpairedDatasetAudit(Dataset):
    """Load preprocessed NPZ slices for one field strength."""
    def __init__(self, base_dir, split, modality, field):
        self.files = sorted((Path(base_dir) / split / modality / field).glob('*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # MUST: Call the audit function
        audit_file_loading(file_path)
        return {'image': npz_to_tensor(np.load(file_path)['image'])}


class PairedDatasetAudit(Dataset):
    """Load paired source–target slices matched by subject + slice index."""
    def __init__(self, base_dir, split, modality, source_field, target_field):
        src_dir = Path(base_dir) / split / modality / source_field
        tgt_dir = Path(base_dir) / split / modality / target_field
        src_map = {self._key(f): f for f in sorted(src_dir.glob('*.npz'))}
        tgt_map = {self._key(f): f for f in sorted(tgt_dir.glob('*.npz'))}
        keys = sorted(set(src_map) & set(tgt_map))
        self.pairs = [(src_map[k], tgt_map[k]) for k in keys]

    @staticmethod
    def _key(p):
        # filename: {split}_{modality}_{field}_{ID}_s{NNN}.npz  →  key '{ID}_s{NNN}'
        parts = p.stem.split('_')
        return '_'.join(parts[-2:])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, t = self.pairs[idx]
        # MUST: Call the audit function
        audit_file_loading(s), audit_file_loading(t)
        return {'source': npz_to_tensor(np.load(s)['image']),
                'target': npz_to_tensor(np.load(t)['image'])}

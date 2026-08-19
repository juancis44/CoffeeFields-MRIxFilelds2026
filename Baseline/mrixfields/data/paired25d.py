"""RAM-cached, 2.5D, paired dataset for supervised restoration (Task 2, Fase 2).

Why this exists (vs. the baseline CachedPairedDataset)
------------------------------------------------------
    * 2.5D input  : source = center slice + K neighbors as channels -> cheap 3D
                    context, sharper/consistent output, negligible extra memory.
    * RAM cache   : with ~500 GB RAM the whole slice set fits in memory; I/O
                    stops being the bottleneck (the CycleGAN run is I/O + single
                    GPU bound). Set cache=True.
    * bbox crop   : crop the MNI brain box (see brain_bbox.py) -> less compute,
                    honest SSIM/nRMSE (no free background).
    * augmentation: consistent flips/rot90 across the source stack AND target.
    * synthetic   : SyntheticLowfieldPaired25D degrades high-field slices into
                    0.1T-like inputs for pre-training (see degrade.py).

Output convention: float32 tensors scaled to [-1, 1] (matches the baseline
generators' Tanh range). Source shape (2K+1, H, W); target shape (1, H, W).

Torch is imported lazily so the pure-numpy assembly logic can be unit-tested
without a torch install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .brain_bbox import apply_bbox
from .degrade import DegradeConfig, simulate_lowfield


# --------------------------------------------------------------------------- #
#  Filename parsing / indexing (pure)
# --------------------------------------------------------------------------- #
def parse_npz_name(path: Path) -> Tuple[str, int]:
    """`..._{prefix}_{id}_s{NNN}.npz` -> (subject='prefix_id', slice_idx=NNN)."""
    parts = path.stem.split("_")
    slice_idx = int(parts[-1][1:])           # 's128' -> 128
    subject = "_".join(parts[-3:-1])         # 'P_0006'
    return subject, slice_idx


def index_by_subject(files: List[Path]) -> Dict[str, Dict[int, Path]]:
    idx: Dict[str, Dict[int, Path]] = {}
    for f in files:
        subj, s = parse_npz_name(f)
        idx.setdefault(subj, {})[s] = f
    return idx


def neighbor_slices(center: int, available: Dict[int, Path], k: int) -> List[int]:
    """Return 2k+1 slice indices around `center`, clamped to the nearest existing."""
    if not available:
        return [center] * (2 * k + 1)
    lo, hi = min(available), max(available)
    out = []
    for off in range(-k, k + 1):
        want = center + off
        want = max(lo, min(hi, want))
        if want not in available:  # nearest existing (handles gaps)
            want = min(available, key=lambda a: abs(a - want))
        out.append(want)
    return out


# --------------------------------------------------------------------------- #
#  Augmentation (pure numpy, consistent across a list of 2D arrays)
# --------------------------------------------------------------------------- #
def augment_arrays(arrays: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
    """Apply the SAME random flips / 180-deg rotation to every array in the list.

    Rotation is restricted to 0 or 180 degrees (k in {0, 2}); 90-deg rotations
    transpose the array and would change (H, W) on non-square brain crops,
    breaking batch collation.
    """
    hflip = rng.random() < 0.5
    vflip = rng.random() < 0.5
    krot = 2 * int(rng.integers(0, 2))       # 0 or 180 deg -> shape preserved
    out = []
    for a in arrays:
        if hflip:
            a = a[:, ::-1]
        if vflip:
            a = a[::-1, :]
        if krot:
            a = np.rot90(a, k=krot)
        out.append(np.ascontiguousarray(a))
    return out


def _to_tensor(stack: np.ndarray):
    import torch
    t = torch.from_numpy(stack.astype(np.float32))
    return t * 2.0 - 1.0            # [0,1] -> [-1,1]


# --------------------------------------------------------------------------- #
#  Real paired dataset
# --------------------------------------------------------------------------- #
class RAMCachedPaired25D:
    """Paired source/target slices with 2.5D input, RAM cache, bbox crop, aug.

    Args:
        preprocessed_dir: root of extracted npz (PREPROCESSED_DIR).
        split:            e.g. 'pro_train' (paired prospective).
        modality:         'T1W' | 'T2W' | 'T2FLAIR'.
        source_field:     '0.1T'.
        target_field:     '1.5T' | '3T' | '5T' | '7T'.
        bbox:             dict from brain_bbox (None = no crop).
        n_neighbors:      K; source has 2K+1 channels.
        augment:          random flips/rot90 (train only).
        cache:            preload all arrays into RAM.
    """

    def __init__(
        self,
        preprocessed_dir,
        split: str,
        modality: str,
        source_field: str,
        target_field: str,
        bbox: Optional[Dict] = None,
        n_neighbors: int = 1,
        augment: bool = True,
        cache: bool = True,
        seed: int = 0,
        subjects=None,
        pair_by: str = "id",
    ):
        self.k = int(n_neighbors)
        self.bbox = bbox
        self.augment = augment
        self.cache = cache
        self._rng = np.random.default_rng(seed)

        base = Path(preprocessed_dir)
        src_files = sorted((base / split / modality / source_field).glob("*.npz"))
        tgt_files = sorted((base / split / modality / target_field).glob("*.npz"))
        if not src_files or not tgt_files:
            raise FileNotFoundError(
                f"Missing npz for {source_field}/{target_field} under "
                f"{base / split / modality}. Run preprocess.py extract-slices first."
            )

        self.src_idx = index_by_subject(src_files)
        self.tgt_idx = index_by_subject(tgt_files)

        # map each source subject -> its target subject.
        #   pair_by="id"    : same subject id at both fields (default).
        #   pair_by="order" : i-th source subject <-> i-th target subject (sorted),
        #                     for paired sets where the two fields use different ids
        #                     (e.g. Validating_prospective: 0.1T P_0001.. vs 3T P_0010..).
        if pair_by == "order":
            src_subs, tgt_subs = sorted(self.src_idx), sorted(self.tgt_idx)
            n = min(len(src_subs), len(tgt_subs))
            self.tgt_of = {src_subs[i]: tgt_subs[i] for i in range(n)}
        else:
            self.tgt_of = {sj: sj for sj in self.src_idx if sj in self.tgt_idx}

        # a sample = (source_subject, center_slice) present in BOTH source and mapped target
        self.samples: List[Tuple[str, int]] = []
        for subj, sl_map in self.src_idx.items():
            tgt_subj = self.tgt_of.get(subj)
            if tgt_subj is None:
                continue
            tgt_map = self.tgt_idx.get(tgt_subj, {})
            for s in sorted(sl_map):
                if s in tgt_map:
                    self.samples.append((subj, s))
        if subjects is not None:
            keep = set(subjects)
            self.samples = [(sj, sl) for (sj, sl) in self.samples if sj in keep]
        if not self.samples:
            raise ValueError("No overlapping (subject, slice) pairs found.")

        self._mem: Dict[Path, np.ndarray] = {}
        if cache:
            for f in src_files + tgt_files:
                self._mem[f] = np.load(f)["image"].astype(np.float32)

    # ---- pure numpy core (unit-testable without torch) ------------------- #
    def _load(self, path: Path) -> np.ndarray:
        if path in self._mem:
            return self._mem[path]
        arr = np.load(path)["image"].astype(np.float32)
        if self.cache:
            self._mem[path] = arr
        return arr

    def get_numpy(self, index: int, rng: Optional[np.random.Generator] = None):
        """Return (source[C,H,W], target[1,H,W]) as float32 numpy in [0,1]."""
        rng = rng or self._rng
        subj, center = self.samples[index]
        src_map = self.src_idx[subj]
        idxs = neighbor_slices(center, src_map, self.k)

        src_slices = [self._load(src_map[i]) for i in idxs]
        tgt_subj = self.tgt_of[subj]
        tgt_slice = self._load(self.tgt_idx[tgt_subj][center])

        if self.bbox is not None:
            src_slices = [apply_bbox(s, self.bbox) for s in src_slices]
            tgt_slice = apply_bbox(tgt_slice, self.bbox)

        if self.augment:
            aug = augment_arrays(src_slices + [tgt_slice], rng)
            src_slices, tgt_slice = aug[:-1], aug[-1]

        source = np.stack(src_slices, axis=0)           # (C, H, W)
        target = tgt_slice[None]                         # (1, H, W)
        return source, target

    # ---- torch interface -------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        source, target = self.get_numpy(index)
        subj, center = self.samples[index]
        return {
            "source": _to_tensor(source),
            "target": _to_tensor(target),
            "subject_id": subj,
            "slice_idx": center,
        }


# --------------------------------------------------------------------------- #
#  Synthetic paired dataset (pre-training)
# --------------------------------------------------------------------------- #
class SyntheticLowfieldPaired25D:
    """On-the-fly synthetic pairs: target = real high-field slice,
    source = its degraded 0.1T-like version (see degrade.py).

    Draws from abundant retrospective high-field data to pre-train the model
    before fine-tuning on the few real 0.1T pairs.

    Args mirror RAMCachedPaired25D, minus source_field; add `target_field`
    (the high-field slices to degrade) and `degrade_cfg`.
    """

    def __init__(
        self,
        preprocessed_dir,
        split: str,
        modality: str,
        target_field: str,
        bbox: Optional[Dict] = None,
        n_neighbors: int = 1,
        augment: bool = True,
        cache: bool = True,
        degrade_cfg: Optional[DegradeConfig] = None,
        seed: int = 0,
        subjects=None,
    ):
        self.k = int(n_neighbors)
        self.bbox = bbox
        self.augment = augment
        self.cache = cache
        self.cfg = degrade_cfg or DegradeConfig()
        self._rng = np.random.default_rng(seed)

        base = Path(preprocessed_dir)
        files = sorted((base / split / modality / target_field).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No npz under {base / split / modality / target_field}")
        self.idx = index_by_subject(files)
        self.samples: List[Tuple[str, int]] = [
            (subj, s) for subj, m in self.idx.items() for s in sorted(m)
        ]
        if subjects is not None:
            keep = set(subjects)
            self.samples = [(sj, sl) for (sj, sl) in self.samples if sj in keep]
        self._mem: Dict[Path, np.ndarray] = {}
        if cache:
            for f in files:
                self._mem[f] = np.load(f)["image"].astype(np.float32)

    def _load(self, path: Path) -> np.ndarray:
        if path in self._mem:
            return self._mem[path]
        arr = np.load(path)["image"].astype(np.float32)
        if self.cache:
            self._mem[path] = arr
        return arr

    def get_numpy(self, index: int, rng: Optional[np.random.Generator] = None):
        rng = rng or self._rng
        subj, center = self.samples[index]
        smap = self.idx[subj]
        idxs = neighbor_slices(center, smap, self.k)

        clean = [self._load(smap[i]) for i in idxs]      # high-field neighbors
        # degrade the whole stack coherently (one seed) -> realistic source
        deg_seed = int(rng.integers(0, 2**31 - 1))
        source = [simulate_lowfield(c, self.cfg, seed=deg_seed) for c in clean]
        target = clean[self.k]                            # center clean slice

        if self.bbox is not None:
            source = [apply_bbox(s, self.bbox) for s in source]
            target = apply_bbox(target, self.bbox)
        if self.augment:
            aug = augment_arrays(source + [target], rng)
            source, target = aug[:-1], aug[-1]

        return np.stack(source, axis=0), target[None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        source, target = self.get_numpy(index)
        subj, center = self.samples[index]
        return {
            "source": _to_tensor(source),
            "target": _to_tensor(target),
            "subject_id": subj,
            "slice_idx": center,
        }


# --------------------------------------------------------------------------- #
#  Unpaired single-field dataset (Fase 4 — semi-supervised)
# --------------------------------------------------------------------------- #
class Unpaired25D:
    """Single-field 2.5D slices with no target (unpaired retrospective data).

    Two uses in Fase 4:
        * real 0.1T inputs (n_neighbors=K -> 2K+1 channels) fed to G to get an
          extra adversarial + structure signal without paired ground truth,
        * real high-field images (n_neighbors=0 -> 1 channel) as *real* examples
          for the discriminator.

    Returns {"image": tensor(C,H,W) in [-1,1], subject_id, slice_idx}.
    """

    def __init__(self, preprocessed_dir, split, modality, field,
                 bbox=None, n_neighbors=1, augment=True, cache=True, seed=0):
        self.k = int(n_neighbors)
        self.bbox = bbox
        self.augment = augment
        self.cache = cache
        self._rng = np.random.default_rng(seed)

        base = Path(preprocessed_dir)
        files = sorted((base / split / modality / field).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No npz under {base / split / modality / field}")
        self.idx = index_by_subject(files)
        self.samples = [(subj, s) for subj, m in self.idx.items() for s in sorted(m)]
        self._mem = {}
        if cache:
            for f in files:
                self._mem[f] = np.load(f)["image"].astype(np.float32)

    def _load(self, path):
        if path in self._mem:
            return self._mem[path]
        arr = np.load(path)["image"].astype(np.float32)
        if self.cache:
            self._mem[path] = arr
        return arr

    def get_numpy(self, index, rng=None):
        rng = rng or self._rng
        subj, center = self.samples[index]
        smap = self.idx[subj]
        idxs = neighbor_slices(center, smap, self.k)
        slices = [self._load(smap[i]) for i in idxs]
        if self.bbox is not None:
            slices = [apply_bbox(s, self.bbox) for s in slices]
        if self.augment:
            slices = augment_arrays(slices, rng)
        return np.stack(slices, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.get_numpy(index)
        subj, center = self.samples[index]
        return {"image": _to_tensor(img), "subject_id": subj, "slice_idx": center}


# module loads cleanly without torch (torch only used inside _to_tensor)

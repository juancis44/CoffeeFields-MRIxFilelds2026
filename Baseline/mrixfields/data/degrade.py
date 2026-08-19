"""Ultra-low-field (0.1T) degradation simulator for Task 2.

Purpose
-------
`Training_prospective` (paired travelling volunteers) is small. But
`Training_retrospective` contains many high-field volumes. By degrading a
high-field slice into a *0.1T-like* one we synthesize extra source/target
pairs to **pre-train** a supervised restoration model, then fine-tune on the
few real 0.1T->target pairs.

What 0.1T actually costs you (and what we model here):
    - Low SNR                -> additive Rician noise
    - Low spatial resolution -> Gaussian blur + down/up-sample (partial volume)
    - B0/B1 inhomogeneity    -> smooth multiplicative bias field
    - Altered contrast       -> gamma / mild intensity remap

Caveat
------
Real 0.1T contrast differs from high field for *relaxometric* reasons
(T1/T2 change with field strength) that a spatial degradation cannot fully
reproduce. This simulator is intended for **pre-training / domain
randomization**, not as a replacement for the real paired data. Always
fine-tune on the real pairs afterwards.

All functions operate on float32 arrays in [0, 1] (the challenge convention)
and return float32 in [0, 1]. Works on 2D slices or 3D volumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class DegradeConfig:
    """Ranges for randomized 0.1T-like degradation (domain randomization).

    Each call samples uniformly within the [lo, hi] range so the network sees
    a *distribution* of degradations rather than one fixed operator.
    """
    blur_sigma: Tuple[float, float] = (0.8, 1.8)        # px, Gaussian PSF
    downsample: Tuple[float, float] = (2.0, 4.0)        # resolution loss factor
    rician_noise: Tuple[float, float] = (0.02, 0.08)    # noise std (image in [0,1])
    bias_strength: Tuple[float, float] = (0.1, 0.35)    # +/- multiplicative field
    bias_smooth: Tuple[float, float] = (24.0, 48.0)     # px, smoothness of bias
    gamma: Tuple[float, float] = (0.8, 1.3)             # contrast remap
    keep_range: bool = True                             # renormalize to [0,1]
    # Background handling (added after why_separable.py found the real leak).
    # Rician noise computes sqrt((x+n1)^2 + n2^2), which is POSITIVE even where
    # x = 0, so it fills the black background. Measured on real data: 49% of a
    # real 0.1T slice is background at ~0, versus 0.06% in the old synthetic
    # input. The network therefore trained with half the image full of noise and
    # learned denoising instead of super-resolution. Re-zeroing the background
    # after degradation removes the single largest train/test discrepancy.
    kspace_resolution: bool = False     # True = physical k-space truncation
    preserve_background: bool = True
    bg_thresh: float = 0.01                             # background = input <= this
    # Spectrum control (added after the domain-gap diagnosis). Real 0.1T has
    # ~50-120x LESS high-frequency energy than the old simulator produced,
    # because Rician noise was applied AFTER blur and stayed as broadband
    # high-freq. noise_before_blur puts the noise ahead of blur/downsample so
    # the PSF smooths it, matching the low-freq spectrum of real 0.1T; the
    # network then learns to restore blur (super-resolution) rather than to
    # denoise. Default False preserves the already-trained models' recipe.
    noise_before_blur: bool = False
    # (q_grid, ref_quantiles) from histmatch.build_reference() on REAL 0.1T.
    # Carried in the config so it reaches simulate_lowfield through the
    # existing dataset plumbing without changing any dataset signature.
    contrast_ref: object = None


def realistic_preset() -> "DegradeConfig":
    """Moderate spectrum fix + contrast mapping enabled.

    lowfreq_preset() matched the real-0.1T high-frequency energy exactly and made
    things WORSE (subcortical structures collapsed): the input became so degraded
    that the net learned to output blur instead of learning super-resolution.
    This preset backs off to roughly half that degradation while keeping the
    structural correction (noise before blur), and relies on `contrast_ref` to
    close the intensity gap -- which the domain diagnosis measured at W1
    0.15-0.32 for EVERY field, i.e. a large uniform mismatch we never fixed."""
    return DegradeConfig(
        blur_sigma=(1.1, 2.2),
        downsample=(2.5, 4.5),
        rician_noise=(0.015, 0.05),
        bias_strength=(0.1, 0.35),
        bias_smooth=(24.0, 48.0),
        gamma=(0.9, 1.15),          # narrower: contrast now handled by the quantile map
        keep_range=True,
        noise_before_blur=True,
    )


def apply_contrast_map(x: np.ndarray, contrast_ref, brain_thresh: float = 0.05) -> np.ndarray:
    """Map brain intensities onto the REAL 0.1T population distribution.

    `contrast_ref` is (q_grid, ref_quantiles) from
    mrixfields.data.histmatch.build_reference() pointed at real 0.1T volumes.
    This is the missing piece the old simulator never modelled: relaxometric
    contrast differs between fields, and blur+noise alone cannot reproduce it,
    so the synthetic input kept its high-field contrast. The map is monotone, so
    it changes the intensity distribution without moving any anatomy.
    """
    if contrast_ref is None:
        return x.astype(np.float32)
    from .histmatch import match_histogram
    q_grid, ref_q = contrast_ref
    mask = x > brain_thresh
    if not mask.any():
        return x.astype(np.float32)
    return match_histogram(x.astype(np.float32), mask, q_grid, ref_q).astype(np.float32)


def physical_preset() -> "DegradeConfig":
    """Everything the diagnostics pointed to: background preserved, k-space
    resolution loss (no interpolation fingerprint), noise before blur, and a
    contrast map supplied at call time via contrast_ref."""
    return DegradeConfig(
        blur_sigma=(0.9, 1.8),
        downsample=(2.0, 3.5),
        rician_noise=(0.015, 0.05),
        bias_strength=(0.1, 0.3),
        bias_smooth=(24.0, 48.0),
        gamma=(0.95, 1.1),
        keep_range=True,
        kspace_resolution=True,
        preserve_background=True,
        noise_before_blur=True,
    )


def lowfreq_preset() -> "DegradeConfig":
    """Spectrum-matched degradation: heavier blur/resolution loss, noise moved
    ahead of the blur, lower noise std. Produces a 0.1T-like input whose
    high-frequency energy is close to real 0.1T (calibrate with
    scripts/calibrate_degrade.py). Use this to RE-train, not to reproduce the
    current submission."""
    return DegradeConfig(
        blur_sigma=(1.6, 3.2),
        downsample=(3.0, 6.0),
        rician_noise=(0.008, 0.03),
        bias_strength=(0.1, 0.35),
        bias_smooth=(24.0, 48.0),
        gamma=(0.8, 1.3),
        keep_range=True,
        noise_before_blur=True,
    )


# --------------------------------------------------------------------------- #
#  Primitive operators
# --------------------------------------------------------------------------- #
def _rng(seed):
    return np.random.default_rng(seed)


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def apply_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(x, sigma=sigma).astype(np.float32)


def apply_resolution_loss(x: np.ndarray, factor: float) -> np.ndarray:
    """Downsample then upsample back to the original grid (partial-volume blur)."""
    if factor <= 1.0:
        return x.astype(np.float32)
    down = 1.0 / factor
    small = zoom(x, down, order=1)
    small = np.clip(small, 0, None)
    # zoom back to exact original shape
    zoom_back = tuple(o / s for o, s in zip(x.shape, small.shape))
    up = zoom(small, zoom_back, order=1)
    # guard against off-by-one from rounding
    up = _match_shape(up, x.shape)
    return up.astype(np.float32)



def apply_kspace_truncation(x: np.ndarray, factor: float) -> np.ndarray:
    """Resolution loss the way MRI actually does it: acquire less k-space.

    why_separable.py found the radial-FFT features separate real from synthetic
    at AUC 1.0 even after the background and contrast were matched. The cause is
    apply_resolution_loss's zoom(down)->zoom(up) with linear interpolation, which
    stamps a distinctive spectral fingerprint on every synthetic slice. A real
    low-resolution acquisition instead samples a smaller k-space region, so the
    honest operator is: FFT -> keep the central 1/factor of k-space -> iFFT.
    This also reproduces the Gibbs ringing that real low-field images show and
    interpolation does not.
    """
    if factor <= 1.0:
        return x.astype(np.float32)
    F = np.fft.fftshift(np.fft.fftn(x))
    keep = np.zeros(x.shape, dtype=bool)
    sl = []
    for n in x.shape:
        half = max(1, int(round(n / factor / 2)))
        c = n // 2
        sl.append(slice(max(0, c - half), min(n, c + half)))
    keep[tuple(sl)] = True
    F = F * keep
    out = np.real(np.fft.ifftn(np.fft.ifftshift(F)))
    return np.clip(out, 0.0, None).astype(np.float32)


def apply_rician_noise(x: np.ndarray, std: float, rng) -> np.ndarray:
    """Rician noise = magnitude of a complex signal with Gaussian noise on both channels."""
    if std <= 0:
        return x.astype(np.float32)
    n1 = rng.normal(0.0, std, size=x.shape)
    n2 = rng.normal(0.0, std, size=x.shape)
    out = np.sqrt((x + n1) ** 2 + n2 ** 2)
    return out.astype(np.float32)


def apply_bias_field(x: np.ndarray, strength: float, smooth: float, rng) -> np.ndarray:
    """Smooth multiplicative low-frequency field ~ (1 +/- strength)."""
    if strength <= 0:
        return x.astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=x.shape)
    field_ = gaussian_filter(noise, sigma=smooth)
    # normalize field to [-1, 1] then to multiplicative gain
    fmax = np.abs(field_).max()
    if fmax > 1e-8:
        field_ = field_ / fmax
    gain = 1.0 + strength * field_
    return (x * gain).astype(np.float32)


def apply_gamma(x: np.ndarray, gamma: float) -> np.ndarray:
    x = np.clip(x, 0, None)
    m = x.max()
    if m <= 1e-8:
        return x.astype(np.float32)
    return (m * (x / m) ** gamma).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _match_shape(x: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Center-crop or edge-pad x to exactly `shape`."""
    out = np.zeros(shape, dtype=x.dtype)
    src, dst = [], []
    for s, t in zip(x.shape, shape):
        if s >= t:
            a = (s - t) // 2
            src.append(slice(a, a + t)); dst.append(slice(None))
        else:
            a = (t - s) // 2
            src.append(slice(None)); dst.append(slice(a, a + s))
    out[tuple(dst)] = x[tuple(src)]
    return out


def _renorm01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    m = x.max()
    return (x / m).astype(np.float32) if m > 1e-8 else x.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def _finalize(x, bg, cfg, contrast_ref):
    """Restore the background and rescale, without undoing the contrast map.

    _renorm01 divides by the max, which forces every synthetic slice to max=1.0
    (real 0.1T slices average ~0.75) and partly cancels apply_contrast_map. When
    a contrast reference is in play we only clip, keeping the mapped scale.
    """
    x = np.clip(x, 0.0, None).astype(np.float32)
    if bg is not None:
        x[bg] = 0.0                      # re-zero the background Rician noise filled in
    if contrast_ref is not None:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return _renorm01(x) if cfg.keep_range else x


def simulate_lowfield(
    x: np.ndarray,
    cfg: DegradeConfig | None = None,
    seed: int | None = None,
    contrast_ref=None,
) -> np.ndarray:
    """Degrade a high-field slice/volume into a 0.1T-like image.

    Args:
        x:    float32 array in [0, 1] (2D slice or 3D volume).
        cfg:  DegradeConfig with sampling ranges (defaults if None).
        seed: optional int for reproducibility.

    Returns:
        float32 array in [0, 1], same shape as x.
    """
    cfg = cfg or DegradeConfig()
    if contrast_ref is None:
        contrast_ref = getattr(cfg, "contrast_ref", None)
    rng = _rng(seed)
    x = x.astype(np.float32)

    # Remember where the background is BEFORE degrading: Rician noise turns
    # zeros into positive values and would otherwise erase it.
    bg = (x <= cfg.bg_thresh) if cfg.preserve_background else None

    # Contrast first: remap high-field intensities onto the real 0.1T
    # distribution, THEN apply the acquisition degradation on top. Order matters
    # -- the quantile map must see clean tissue intensities, not noise.
    if contrast_ref is not None:
        x = apply_contrast_map(x, contrast_ref)

    x = apply_gamma(x, _u(rng, cfg.gamma))
    if cfg.noise_before_blur:
        # noise first -> the blur/resolution loss smooths it, so the synthetic
        # input ends up low-frequency like real 0.1T (see lowfreq_preset).
        x = apply_rician_noise(x, _u(rng, cfg.rician_noise), rng)
        x = apply_blur(x, _u(rng, cfg.blur_sigma))
        _res = apply_kspace_truncation if cfg.kspace_resolution else apply_resolution_loss
        x = _res(x, _u(rng, cfg.downsample))
        x = apply_bias_field(x, _u(rng, cfg.bias_strength), _u(rng, cfg.bias_smooth), rng)
        return _finalize(x, bg, cfg, contrast_ref)
    x = apply_blur(x, _u(rng, cfg.blur_sigma))
    x = (apply_kspace_truncation if cfg.kspace_resolution else apply_resolution_loss)(
        x, _u(rng, cfg.downsample))
    x = apply_bias_field(x, _u(rng, cfg.bias_strength), _u(rng, cfg.bias_smooth), rng)
    x = apply_rician_noise(x, _u(rng, cfg.rician_noise), rng)

    return _finalize(x, bg, cfg, contrast_ref)

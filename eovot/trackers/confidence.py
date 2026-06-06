"""Peak-to-Sidelobe Ratio (PSR) confidence estimation for correlation filters.

PSR is a response-map quality metric from Bolme et al. (2010) that measures
how sharply peaked the correlation response is relative to background noise.
High PSR indicates the filter found a strong, localised match; low PSR
signals that the response is flat or ambiguous — a reliable proxy for tracking
failure without requiring ground-truth annotations.

    PSR = (peak - μ_sidelobe) / σ_sidelobe

where the sidelobe region excludes an (exclusion_size × exclusion_size) window
centred on the peak.

Empirical thresholds (Bolme et al. 2010):
    PSR > 7   — confident track (strong peak)
    3 < PSR ≤ 7 — uncertain / degraded conditions
    PSR ≤ 3  — likely failure

Reference
---------
Bolme, D. S. et al. "Visual object tracking using adaptive correlation filters."
IEEE CVPR 2010, pp. 2544-2550.
"""

from __future__ import annotations

import numpy as np


def compute_psr(response: np.ndarray, exclusion_size: int = 11) -> float:
    """Compute the Peak-to-Sidelobe Ratio of a correlation response map.

    Args:
        response:       2-D float array of shape ``(H, W)``.
        exclusion_size: Side length (pixels) of the window centred on the peak
                        that is excluded from the sidelobe statistics.
                        Must be odd and positive.  Default: ``11``.

    Returns:
        PSR as a non-negative float.  Returns ``0.0`` if the sidelobe region
        is empty (response map smaller than the exclusion window).

    Raises:
        ValueError: If *response* is not a 2-D array.
    """
    if response.ndim != 2:
        raise ValueError(
            f"response must be a 2-D array, got shape {response.shape}"
        )

    peak_val = float(response.max())
    peak_y, peak_x = np.unravel_index(np.argmax(response), response.shape)

    half = exclusion_size // 2
    mask = np.ones(response.shape, dtype=bool)
    y_lo = max(0, peak_y - half)
    y_hi = min(response.shape[0], peak_y + half + 1)
    x_lo = max(0, peak_x - half)
    x_hi = min(response.shape[1], peak_x + half + 1)
    mask[y_lo:y_hi, x_lo:x_hi] = False

    sidelobe = response[mask]
    if sidelobe.size == 0:
        return 0.0

    mu = float(sidelobe.mean())
    sigma = float(sidelobe.std())

    # When sigma ≈ 0 the sidelobe is perfectly flat.
    # If the peak exceeds the flat background the PSR is effectively infinite;
    # return a large finite value so callers can still threshold on it.
    if sigma < 1e-10:
        return 100.0 if peak_val > mu + 1e-10 else 0.0

    return max(0.0, (peak_val - mu) / sigma)


def psr_to_confidence(psr: float, low: float = 3.0, high: float = 7.0) -> float:
    """Map a PSR value to a normalised confidence score in ``[0, 1]``.

    Uses a linear ramp between *low* and *high*:

        confidence = clip((psr - low) / (high - low), 0, 1)

    Args:
        psr:  Raw PSR value (non-negative float).
        low:  PSR at which confidence is considered 0.  Default: ``3.0``.
        high: PSR at which confidence saturates to 1.  Default: ``7.0``.

    Returns:
        Float in ``[0, 1]``.
    """
    if high <= low:
        raise ValueError(f"high ({high}) must be greater than low ({low})")
    return float(np.clip((psr - low) / (high - low), 0.0, 1.0))

"""Intent gate: decides when the user *means* to capture a thought.

Real speech BCIs need a deliberate "on switch" so private inner speech does not
leak out (the Stanford inner-speech study used a mental password for this).
Here the switch is a sustained rise in a focus index computed from band power:

    focus = beta / (theta + alpha + beta)

The gate calibrates a resting baseline from the first few windows (median and
MAD, so a stray blip doesn't ruin it), then opens when focus stays above
``baseline + max(k * spread, min_delta)`` for ``min_windows`` consecutive
windows, and closes again when it drops back down. Each open->close cycle is one
"capture episode" that gets sent to the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def band_powers(data: np.ndarray, sampling_rate: int) -> dict[str, float]:
    """Average relative band power across channels (simple periodogram)."""
    data = data - data.mean(axis=1, keepdims=True)
    n = data.shape[1]
    hann = np.hanning(n)
    spec = np.abs(np.fft.rfft(data * hann, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    total = spec[:, (freqs >= 1) & (freqs <= 45)].sum(axis=1) + 1e-12
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float((spec[:, mask].sum(axis=1) / total).mean())
    return out


def focus_index(powers: dict[str, float]) -> float:
    denom = powers["theta"] + powers["alpha"] + powers["beta"] + 1e-12
    return powers["beta"] / denom


@dataclass
class GateEvent:
    kind: str  # "open" | "close"
    t: float
    focus: float


@dataclass
class IntentGate:
    calibration_windows: int = 8
    k: float = 4.0
    min_windows: int = 2
    # Floor on (threshold - baseline). Keeps a very flat baseline from making
    # the gate twitchy.
    min_delta: float = 0.15

    is_open: bool = field(default=False, init=False)
    threshold: float | None = field(default=None, init=False)
    _calib: list[float] = field(default_factory=list, init=False)
    _above: int = field(default=0, init=False)
    _below: int = field(default=0, init=False)

    def update(self, focus: float, t: float) -> GateEvent | None:
        """Feed one focus value; returns an event when the gate changes state."""
        if self.threshold is None:
            self._calib.append(focus)
            if len(self._calib) >= self.calibration_windows:
                # Robust statistics: one stray blip during calibration should
                # not push the threshold out of reach.
                calib = np.asarray(self._calib)
                base = float(np.median(calib))
                spread = 1.4826 * float(np.median(np.abs(calib - base)))
                self.threshold = base + max(self.k * spread, self.min_delta)
            return None

        if not self.is_open:
            self._above = self._above + 1 if focus > self.threshold else 0
            if self._above >= self.min_windows:
                self.is_open = True
                self._above = 0
                return GateEvent("open", t, focus)
        else:
            self._below = self._below + 1 if focus <= self.threshold else 0
            if self._below >= self.min_windows:
                self.is_open = False
                self._below = 0
                return GateEvent("close", t, focus)
        return None

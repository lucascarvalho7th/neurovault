"""Signal sources: where the (simulated or real) brain activity comes from.

A source yields fixed-length windows of multichannel EEG, shaped
(n_channels, n_samples), plus the sampling rate.

Two implementations:

* ``SimulatedSource`` - pure numpy, deterministic, no hardware. It produces an
  alpha-dominant "resting" signal and injects beta-band bursts at scheduled
  moments, which stand in for "the user deliberately starts thinking a note".
  This is the default because it lets the whole pipeline be tested end to end.

* ``BrainFlowSource`` - uses the BrainFlow library. With ``board_id=-1`` it uses
  BrainFlow's synthetic board; with a real board id (OpenBCI Cyton, Muse, ...)
  and connection params it reads a real headset. The rest of the pipeline does
  not change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence

import numpy as np


@dataclass
class Window:
    """One chunk of EEG data."""

    data: np.ndarray  # (n_channels, n_samples), microvolts
    sampling_rate: int
    t_start: float  # seconds since the session started
    # Ground truth from the simulator, if known (None for real hardware).
    # Only used for evaluation/metrics, never by the pipeline itself.
    true_intent: bool | None = None


class SignalSource(Protocol):
    sampling_rate: int

    def windows(self) -> Iterator[Window]: ...

    def close(self) -> None: ...


@dataclass
class SimulatedSource:
    """Synthetic EEG with controllable "intent" bursts.

    Parameters
    ----------
    duration_s:
        Total session length in seconds.
    intent_onsets_s:
        Times (seconds) at which a burst of focused activity starts.
    intent_length_s:
        How long each burst lasts.
    realtime:
        If True, sleep so windows arrive at wall-clock speed (nice for demos).
    """

    duration_s: float = 60.0
    intent_onsets_s: Sequence[float] = (8.0, 20.0, 32.0, 44.0)
    intent_length_s: float = 3.0
    sampling_rate: int = 250
    n_channels: int = 8
    window_s: float = 1.0
    seed: int = 7
    blip_probability: float = 0.08
    realtime: bool = False
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _in_intent(self, t: float) -> bool:
        return any(on <= t < on + self.intent_length_s for on in self.intent_onsets_s)

    def _make_window(self, t0: float) -> np.ndarray:
        n = int(self.sampling_rate * self.window_s)
        t = t0 + np.arange(n) / self.sampling_rate
        mid = t0 + self.window_s / 2
        intent = self._in_intent(mid)

        # Per-window state jitter, so resting focus is not perfectly flat.
        state = self._rng.uniform(0.85, 1.15)
        # Occasional one-window "distraction" blips of beta that are NOT intent.
        # A good gate should ignore these (they never last 2+ windows).
        blip = (not intent) and self._rng.random() < self.blip_probability

        chans = []
        for c in range(self.n_channels):
            phase = self._rng.uniform(0, 2 * np.pi, size=4)
            # Resting state: strong alpha (10 Hz), some theta (6 Hz), weak beta.
            # Focused/intent state: alpha suppressed, beta up (a real, if
            # simplified, EEG signature of active mental effort).
            alpha_amp = (9.0 if intent else 18.0) * state
            beta_amp = (9.0 if intent or blip else 3.5) / state
            sig = (
                alpha_amp * np.sin(2 * np.pi * 10 * t + phase[0])
                + 6.0 * np.sin(2 * np.pi * 6 * t + phase[1])
                + beta_amp * np.sin(2 * np.pi * 20 * t + phase[2])
                + beta_amp * 0.6 * np.sin(2 * np.pi * 24 * t + phase[3])
            )
            # Pink-ish background noise.
            noise = np.cumsum(self._rng.normal(0, 1.0, n))
            noise -= np.linspace(noise[0], noise[-1], n)  # detrend
            sig = sig + 0.5 * noise + self._rng.normal(0, 4.0, n)
            chans.append(sig)
        return np.vstack(chans)

    def windows(self) -> Iterator[Window]:
        n_windows = int(self.duration_s / self.window_s)
        for i in range(n_windows):
            t0 = i * self.window_s
            if self.realtime:
                time.sleep(self.window_s)
            yield Window(
                data=self._make_window(t0),
                sampling_rate=self.sampling_rate,
                t_start=t0,
                true_intent=self._in_intent(t0 + self.window_s / 2),
            )

    def close(self) -> None:  # nothing to release
        pass


class BrainFlowSource:
    """Reads EEG through BrainFlow (synthetic board by default).

    ``board_id=-1`` is BrainFlow's SYNTHETIC_BOARD. Real boards need the
    matching connection parameters, e.g. ``serial_port="COM3"`` for an OpenBCI
    Cyton on Windows. See https://brainflow.readthedocs.io for the full list.
    """

    def __init__(
        self,
        board_id: int = -1,
        duration_s: float = 30.0,
        window_s: float = 1.0,
        serial_port: str = "",
        mac_address: str = "",
    ) -> None:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams

        BoardShim.disable_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address

        self._BoardShim = BoardShim
        self.board_id = board_id
        self.board = BoardShim(board_id, params)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.duration_s = duration_s
        self.window_s = window_s
        self._started = False

    def windows(self) -> Iterator[Window]:
        self.board.prepare_session()
        self.board.start_stream()
        self._started = True
        n_samples = int(self.sampling_rate * self.window_s)
        n_windows = int(self.duration_s / self.window_s)
        try:
            # Let the buffer fill once before the first read.
            time.sleep(self.window_s)
            for i in range(n_windows):
                data = self.board.get_current_board_data(n_samples)
                eeg = data[self.eeg_channels, :]
                if eeg.shape[1] < n_samples:
                    time.sleep(self.window_s)
                    continue
                yield Window(data=eeg, sampling_rate=self.sampling_rate, t_start=i * self.window_s)
                time.sleep(self.window_s)
        finally:
            self.close()

    def close(self) -> None:
        if self._started:
            try:
                self.board.stop_stream()
            finally:
                self.board.release_session()
                self._started = False

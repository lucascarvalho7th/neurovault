"""The capture loop that wires every stage together.

    SignalSource -> IntentGate -> Decoder -> Organizer -> Vault

Nothing here knows whether the signal is simulated or the decoder is a mock,
which is the point: swap one stage, keep the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .decoder import Decoder
from .intent_gate import IntentGate, band_powers, focus_index
from .organizer import Organizer
from .report import render_report
from .signal_source import SignalSource, Window
from .vault import Vault

Logger = Callable[[str], None]


@dataclass
class CapturedNote:
    t_open: float
    t_close: float
    path: Path
    title: str
    type: str
    decoded: str
    intended: str | None
    confidence: float
    cleaned: str = ""
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    organizer: str = ""


@dataclass
class SessionResult:
    notes: list[CapturedNote] = field(default_factory=list)
    episodes: int = 0
    skipped: int = 0
    gate_threshold: float | None = None
    # Only filled when the source knows the ground truth (simulation).
    true_intents: int = 0  # intent bursts that happened after calibration
    caught_intents: int = 0  # bursts that ended up inside a capture episode
    false_episodes: int = 0  # episodes the gate opened with no real intent
    vault_root: Path | None = None

    def summary(self) -> dict:
        d = {
            "organizer": self.notes[0].organizer if self.notes else None,
            "episodes": self.episodes,
            "notes_written": len(self.notes),
            "skipped_episodes": self.skipped,
            "gate_threshold": round(self.gate_threshold, 4) if self.gate_threshold else None,
            "notes": [
                {
                    "title": n.title,
                    "type": n.type,
                    "path": n.path.relative_to(self.vault_root).as_posix() if self.vault_root else str(n.path),
                    "window": [n.t_open, n.t_close],
                    "decoder_confidence": n.confidence,
                    "decoded": n.decoded,
                    "cleaned": n.cleaned,
                    "intended": n.intended,
                    "tags": n.tags,
                    "links": n.links,
                }
                for n in self.notes
            ],
        }
        if self.true_intents:
            d["gate_recall"] = round(self.caught_intents / self.true_intents, 3)
            d["gate_false_episodes"] = self.false_episodes
        return d


class Pipeline:
    def __init__(
        self,
        source: SignalSource,
        decoder: Decoder,
        organizer: Organizer,
        vault: Vault,
        gate: IntentGate | None = None,
        review_below: float = 0.6,
        discard_below: float = 0.3,
        log: Logger = print,
    ) -> None:
        self.source = source
        self.decoder = decoder
        self.organizer = organizer
        self.vault = vault
        self.gate = gate or IntentGate()
        self.review_below = review_below
        self.discard_below = discard_below
        self.log = log
        self.report_path: Path | None = None

    def _finish_episode(self, episode: list[Window], result: SessionResult) -> None:
        result.episodes += 1
        t_open, t_close = episode[0].t_start, episode[-1].t_start
        truth = [w.true_intent for w in episode]
        if None not in truth:  # simulation: score the gate against ground truth
            if any(truth):
                result.caught_intents += 1
            else:
                result.false_episodes += 1
        thought = self.decoder.decode(episode)
        if thought is None or not thought.text.strip():
            result.skipped += 1
            self.log(f"  [{t_open:5.1f}s] episode had nothing to decode, skipped")
            return

        self.log(f"  [{t_open:5.1f}s] decoded ({thought.confidence:.2f}): {thought.text}")
        if thought.confidence < self.discard_below:
            result.skipped += 1
            self.log("           -> discarded (confidence too low, probably not a real thought)")
            return
        note = self.organizer.organize(thought, self.vault.titles())
        if thought.confidence < self.review_below and "needs-review" not in note.tags:
            note.tags.append("needs-review")
        path = self.vault.write_note(note, thought, when=datetime.now())
        self.log(f"           -> {note.type:<10} {path.relative_to(self.vault.root)}")
        result.notes.append(
            CapturedNote(
                t_open=t_open,
                t_close=t_close,
                path=path,
                title=path.stem,
                type=note.type,
                decoded=thought.text,
                intended=thought.intended,
                confidence=thought.confidence,
                cleaned=note.body,
                tags=list(note.tags),
                links=list(note.links),
                organizer=note.organizer,
            )
        )

    def run(self) -> SessionResult:
        result = SessionResult(vault_root=self.vault.root)
        episode: list[Window] = []
        calibrated = False
        prev_intent = False
        self.log("Calibrating resting baseline (stay relaxed)...")
        try:
            for w in self.source.windows():
                focus = focus_index(band_powers(w.data, w.sampling_rate))
                event = self.gate.update(focus, w.t_start)

                if not calibrated and self.gate.threshold is not None:
                    calibrated = True
                    self.log(f"Baseline set. Gate threshold = {self.gate.threshold:.3f}. Capturing.")

                if w.true_intent and not prev_intent and calibrated:
                    result.true_intents += 1
                prev_intent = bool(w.true_intent)

                if event and event.kind == "open":
                    self.log(f"  [{w.t_start:5.1f}s] gate OPEN  (focus {event.focus:.2f})")
                if self.gate.is_open or (event and event.kind == "close"):
                    episode.append(w)
                if event and event.kind == "close":
                    self._finish_episode(episode, result)
                    episode = []

            if episode:  # session ended mid-thought
                self._finish_episode(episode, result)
        finally:
            self.source.close()

        result.gate_threshold = self.gate.threshold
        summary = result.summary()
        self.vault.write_session(summary)
        if result.notes:
            self.report_path = self.vault.write_report(render_report(summary))
            self.log(f"Before/after report: {self.report_path.relative_to(self.vault.root)}")
        return result

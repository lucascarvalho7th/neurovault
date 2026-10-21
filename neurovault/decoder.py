"""Decoders: turn a captured brain episode into text.

Real brain-to-text decoders (e.g. the Brain-to-Text '24/'25 baselines, Meta's
Brain2Qwerty) map neural activity to phonemes or characters and then use a
language model to produce text. They make mistakes: published word error rates
range from a few percent (implants) to ~40% (non-invasive).

``MockDecoder`` stands in for one of those. It "knows" what the user meant from
a script of thoughts, and corrupts the text at a configurable word error rate
so the downstream organizer has to cope with realistic, imperfect input.

To plug in a real model later, implement the ``Decoder`` protocol: take the
list of EEG windows for an episode and return a ``DecodedThought``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .signal_source import Window


@dataclass
class DecodedThought:
    text: str  # what the decoder produced (possibly with errors)
    confidence: float  # 0..1, decoder's self-reported confidence
    intended: str | None = None  # ground truth, only known in simulation


class Decoder(Protocol):
    def decode(self, episode: Sequence[Window]) -> DecodedThought | None: ...


_VOWELS = "aeiou"


def _corrupt_word(word: str, rng: random.Random) -> str:
    """Phoneme-ish corruption: swap one vowel, or drop a letter."""
    letters = list(word)
    idx = [i for i, ch in enumerate(letters) if ch.lower() in _VOWELS]
    if idx and rng.random() < 0.7:
        i = rng.choice(idx)
        options = [v for v in _VOWELS if v != letters[i].lower()]
        letters[i] = rng.choice(options)
    elif len(letters) > 3:
        del letters[rng.randrange(1, len(letters) - 1)]
    return "".join(letters)


def add_word_errors(text: str, wer: float, rng: random.Random) -> tuple[str, int]:
    """Return (noisy_text, n_errors). Errors: substitution, deletion, repetition."""
    words = text.split()
    out: list[str] = []
    errors = 0
    for w in words:
        if rng.random() >= wer:
            out.append(w)
            continue
        errors += 1
        r = rng.random()
        if r < 0.6:
            out.append(_corrupt_word(w, rng))
        elif r < 0.85:
            continue  # deletion
        else:
            out.extend([w, w])  # stutter / repetition
    return " ".join(out), errors


def load_thoughts(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


class MockDecoder:
    def __init__(self, thoughts: Sequence[str], wer: float = 0.1, seed: int = 7, loop: bool = False):
        if not 0.0 <= wer <= 1.0:
            raise ValueError("wer must be between 0 and 1")
        self.thoughts = list(thoughts)
        self.wer = wer
        self.loop = loop
        self._rng = random.Random(seed)
        self._i = 0

    def _noise(self) -> DecodedThought:
        """What a decoder emits when the gate fired but no thought was formed:
        a few garbled fragments with low confidence."""
        pool = " ".join(self.thoughts).split() or ["uh"]
        words = [_corrupt_word(self._rng.choice(pool), self._rng) for _ in range(self._rng.randint(2, 4))]
        return DecodedThought(text=" ".join(words), confidence=round(self._rng.uniform(0.05, 0.25), 2))

    def decode(self, episode: Sequence[Window]) -> DecodedThought | None:
        if not self.thoughts:
            return None
        truth = [w.true_intent for w in episode]
        if truth and None not in truth and not any(truth):
            # Simulation only: the gate opened on a distraction, not an intent.
            return self._noise()
        if self._i >= len(self.thoughts):
            if not self.loop:
                return None
            self._i = 0
        intended = self.thoughts[self._i]
        self._i += 1

        noisy, errors = add_word_errors(intended, self.wer, self._rng)
        n_words = max(len(intended.split()), 1)
        # Longer, steadier episodes -> a bit more confident. Mostly driven by
        # how many errors slipped in, plus some jitter like a real model.
        observed = 1.0 - errors / n_words
        length_bonus = min(len(episode), 4) * 0.02
        confidence = max(0.0, min(1.0, observed * 0.9 + length_bonus + self._rng.uniform(-0.05, 0.05)))
        return DecodedThought(text=noisy, confidence=round(confidence, 2), intended=intended)

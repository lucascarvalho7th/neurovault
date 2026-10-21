"""Before/after report: what the decoder produced vs. what ended up in the vault.

In simulation we also know what the user *meant*, so the report can measure
how much the organizer repaired: word error rate (WER) of the raw decoder
output vs. WER of the filed note, both against the intended sentence.
"""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[\w]+", re.UNICODE)


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return _WORD.findall(text)


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Levenshtein distance over words, divided by the reference length."""
    hyp, ref = _words(hypothesis), _words(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def _cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def render_report(summary: dict) -> str:
    notes = summary.get("notes", [])
    organizer = summary.get("organizer") or "unknown"
    scored = [n for n in notes if n.get("intended")]

    lines = [
        "---",
        "source: neurovault",
        "type: report",
        "tags: [neurovault, report]",
        "---",
        "# Session report: before and after",
        "",
        f"**Organizer:** `{organizer}`  ",
        f"**Captured:** {summary.get('notes_written', 0)} notes from {summary.get('episodes', 0)} episodes",
    ]
    if "gate_recall" in summary:
        lines[-1] += (
            f"  \n**Intent gate:** {summary['gate_recall']:.0%} of intended thoughts caught, "
            f"{summary.get('gate_false_episodes', 0)} false captures"
        )

    if scored:
        before = sum(word_error_rate(n["decoded"], n["intended"]) for n in scored) / len(scored)
        after = sum(word_error_rate(n.get("cleaned") or n["decoded"], n["intended"]) for n in scored) / len(scored)
        lines += [
            "",
            "## Word error rate",
            "",
            "| | WER vs. intended thought |",
            "|---|---|",
            f"| Before (raw decoder output) | {before:.0%} |",
            f"| After (filed note) | {after:.0%} |",
        ]
        if organizer == "offline":
            lines += ["", "The offline organizer files text as decoded. Run with Claude "
                          "(`--organizer claude-code` or `--organizer claude`) to repair it."]
        elif before > 0:
            change = (before - after) / before
            lines += ["", f"The organizer removed **{change:.0%}** of the decoding errors."]

    lines += [
        "",
        "## Thought by thought",
        "",
        "| # | Before: decoder output | After: filed note | Type | Conf. |",
        "|---|---|---|---|---|",
    ]
    for i, n in enumerate(notes, 1):
        after = f"**[[{n['title']}]]**<br>{_cell(n.get('cleaned') or n['decoded'])}"
        lines.append(
            f"| {i} | {_cell(n['decoded'])} | {after} | {n['type']} | {n['decoder_confidence']:.2f} |"
        )

    if scored:
        lines += ["", "## What was actually meant (simulation ground truth)", ""]
        for i, n in enumerate(notes, 1):
            if n.get("intended"):
                lines.append(f"{i}. {n['intended']}")
    lines.append("")
    return "\n".join(lines)

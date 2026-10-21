"""End-to-end and unit tests. None of them need hardware or an API key."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from neurovault.decoder import DecodedThought, MockDecoder, add_word_errors
from neurovault.intent_gate import IntentGate, band_powers, focus_index
from neurovault.organizer import ClaudeOrganizer, OfflineOrganizer
from neurovault.pipeline import Pipeline
from neurovault.signal_source import SimulatedSource
from neurovault.vault import Vault, safe_filename

THOUGHTS = [
    "What if the vault could suggest which old notes I should revisit",
    "I need to renew the domain before Friday",
    "Why do I have better ideas while walking?",
]


# --------------------------------------------------------------------------- #
# Signal + gate
# --------------------------------------------------------------------------- #

def test_focus_is_higher_during_intent():
    src = SimulatedSource(duration_s=30, intent_onsets_s=(10.0,), intent_length_s=4.0, seed=1)
    rest, focused = [], []
    for w in src.windows():
        f = focus_index(band_powers(w.data, w.sampling_rate))
        (focused if w.true_intent else rest).append(f)
    assert np.mean(focused) > 3 * np.mean(rest)


def test_band_powers_are_relative():
    src = SimulatedSource(duration_s=2, seed=3)
    w = next(iter(src.windows()))
    p = band_powers(w.data, w.sampling_rate)
    assert 0.95 <= sum(p.values()) <= 1.05


def test_gate_ignores_single_window_blips():
    gate = IntentGate(calibration_windows=4, min_windows=2)
    for f in [0.05, 0.06, 0.05, 0.04]:
        gate.update(f, 0)
    assert gate.threshold is not None
    events = [gate.update(f, t) for t, f in enumerate([0.05, 0.9, 0.05, 0.05])]
    assert not any(events)


def test_gate_opens_and_closes_on_sustained_focus():
    gate = IntentGate(calibration_windows=4, min_windows=2)
    for f in [0.05, 0.06, 0.05, 0.04]:
        gate.update(f, 0)
    seq = [0.05, 0.6, 0.7, 0.65, 0.05, 0.05]
    kinds = [e.kind for t, f in enumerate(seq) if (e := gate.update(f, t))]
    assert kinds == ["open", "close"]


def test_gate_recall_over_many_sessions():
    onsets = [10.0 + 12.0 * i for i in range(4)]
    caught = total = 0
    for seed in range(20):
        src = SimulatedSource(duration_s=60, intent_onsets_s=onsets, seed=seed)
        gate = IntentGate()
        opens = []
        for w in src.windows():
            e = gate.update(focus_index(band_powers(w.data, w.sampling_rate)), w.t_start)
            if e and e.kind == "open":
                opens.append(w.t_start)
        caught += sum(any(on <= t <= on + 4 for t in opens) for on in onsets)
        total += len(onsets)
    assert caught / total >= 0.9


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #

def test_zero_wer_is_lossless():
    text, errors = add_word_errors("keep every single word", 0.0, random.Random(0))
    assert text == "keep every single word" and errors == 0


def test_wer_roughly_matches_target():
    rng = random.Random(0)
    sentence = " ".join(["word"] * 50)
    errs = sum(add_word_errors(sentence, 0.2, rng)[1] for _ in range(200))
    assert 0.15 <= errs / (50 * 200) <= 0.25


def test_mock_decoder_emits_noise_for_false_capture():
    dec = MockDecoder(THOUGHTS, wer=0.0, seed=0)
    fake = [SimpleNamespace(true_intent=False)] * 3
    out = dec.decode(fake)
    assert out.intended is None and out.confidence < 0.3


# --------------------------------------------------------------------------- #
# Organizers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,kind",
    [
        ("I need to call the bank tomorrow", "task"),
        ("Preciso renovar o domínio antes de sexta", "task"),
        ("What if notes could link themselves", "idea"),
        ("Decidi usar o Claude para a revisão", "decision"),
        ("Why is focus easier in the morning?", "question"),
        ("Today was a calm and steady day", "reflection"),
    ],
)
def test_offline_classifier(text, kind):
    assert OfflineOrganizer().classify(text) == kind


class _FakeMessages:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="tool_use", name="file_note", input=self.payload)
        return SimpleNamespace(content=[block])


def test_claude_organizer_parses_tool_call_and_filters_links():
    payload = {
        "cleaned_text": "What if the vault could suggest which old notes I should revisit each week",
        "type": "idea",
        "title": "Vault suggests notes to revisit",
        "summary": "A weekly resurfacing feature for old notes.",
        "tags": ["#Obsidian", "review"],
        "links": ["Weekly review", "A title that does not exist"],
    }
    fake = SimpleNamespace(messages=_FakeMessages(payload))
    org = ClaudeOrganizer(model="test-model", client=fake)
    note = org.organize(
        DecodedThought(text="What if the viult cuuld sugget which old notes", confidence=0.6),
        existing_titles=["Weekly review", "Groceries"],
    )
    assert note.title == "Vault suggests notes to revisit"
    assert note.type == "idea"
    assert note.tags == ["obsidian", "review"]
    assert note.links == ["Weekly review"]  # hallucinated link dropped
    call = fake.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "file_note"}
    assert "Weekly review" in call["messages"][0]["content"]


# --------------------------------------------------------------------------- #
# Vault + full pipeline
# --------------------------------------------------------------------------- #

def test_safe_filename_strips_obsidian_unsafe_chars():
    assert safe_filename('What: a [[test]] / "note"?') == "What a test note"


def test_end_to_end_offline(tmp_path):
    onsets = [10.0 + 12.0 * i for i in range(3)]
    pipe = Pipeline(
        source=SimulatedSource(duration_s=50, intent_onsets_s=onsets, seed=2),
        decoder=MockDecoder(THOUGHTS, wer=0.1, seed=2),
        organizer=OfflineOrganizer(),
        vault=Vault(tmp_path),
        log=lambda *_: None,
    )
    result = pipe.run()
    summary = result.summary()

    assert summary["notes_written"] == 3
    assert summary["gate_recall"] == 1.0
    notes = [p for p in tmp_path.rglob("*.md") if p.parent.name not in ("Daily", "Reports")]
    assert len(notes) == 3
    text = notes[0].read_text(encoding="utf-8")
    assert text.startswith("---\ncreated:")
    assert "source: neurovault" in text
    daily = list((tmp_path / "Daily").glob("*.md"))
    assert len(daily) == 1 and daily[0].read_text().count("[[") == 3
    assert list((tmp_path / ".neurovault" / "sessions").glob("*.json"))


def test_existing_vault_notes_are_offered_for_linking(tmp_path):
    (tmp_path / "Projects").mkdir()
    (tmp_path / "Projects" / "Domain renewal checklist.md").write_text("# x")
    vault = Vault(tmp_path)
    assert "Domain renewal checklist" in vault.titles()
    note = OfflineOrganizer().organize(
        DecodedThought(text="I need to renew the domain before Friday", confidence=0.9),
        vault.titles(),
    )
    assert "Domain renewal checklist" in note.links


def test_decision_wins_over_task_words():
    assert OfflineOrganizer().classify("Decidi usar o Claude em vez de fazer na mão") == "decision"


def test_session_summary_uses_vault_relative_paths(tmp_path):
    pipe = Pipeline(
        source=SimulatedSource(duration_s=30, intent_onsets_s=(12.0,), seed=2),
        decoder=MockDecoder(THOUGHTS, wer=0.0, seed=2),
        organizer=OfflineOrganizer(),
        vault=Vault(tmp_path),
        log=lambda *_: None,
    )
    note = pipe.run().summary()["notes"][0]
    assert not note["path"].startswith("/") and note["path"].endswith(".md")


# --------------------------------------------------------------------------- #
# Before/after report
# --------------------------------------------------------------------------- #

from neurovault.report import render_report, word_error_rate  # noqa: E402


def test_word_error_rate():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert word_error_rate("the cot sat", "the cat sat") == pytest.approx(1 / 3)
    assert word_error_rate("Preciso renovar!", "preciso renovar") == 0.0


class _RepairingClient:
    """Fake Claude that 'repairs' text by returning the intended sentence."""

    def __init__(self, fixes):
        self.fixes = fixes
        self.messages = self

    def create(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        decoded = content.split("<decoded>")[1].split("</decoded>")[0]
        fixed = self.fixes.get(decoded, decoded)
        payload = {"cleaned_text": fixed, "type": "idea", "title": fixed[:40],
                   "summary": fixed, "tags": ["test"], "links": []}
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=payload)])


def test_report_shows_errors_removed_by_claude(tmp_path):
    thoughts = ["E se o vault sugerisse notas antigas toda semana"]
    decoder = MockDecoder(thoughts, wer=0.5, seed=4)
    # Pre-compute what the decoder will emit so the fake client can "fix" it.
    probe = MockDecoder(thoughts, wer=0.5, seed=4).decode([SimpleNamespace(true_intent=True)])
    org = ClaudeOrganizer(model="fake", client=_RepairingClient({probe.text: thoughts[0]}))
    pipe = Pipeline(
        source=SimulatedSource(duration_s=30, intent_onsets_s=(12.0,), seed=2),
        decoder=decoder, organizer=org, vault=Vault(tmp_path), log=lambda *_: None,
    )
    pipe.run()
    report = pipe.report_path.read_text(encoding="utf-8")
    assert "After (filed note) | 0% |" in report
    assert "removed **100%**" in report
    assert "claude:fake" in report
    assert "Reports" not in Vault(tmp_path).titles()


# --------------------------------------------------------------------------- #
# Claude Code backend (subprocess faked, no CLI needed)
# --------------------------------------------------------------------------- #

import json as _json  # noqa: E402

from neurovault.organizer import ClaudeCodeError, ClaudeCodeOrganizer  # noqa: E402


class _FakeRun:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        out, err, code = self.responses.pop(0)
        return SimpleNamespace(stdout=out, stderr=err, returncode=code)


def _ok(structured=None, result=""):
    payload = {"is_error": False, "subtype": "success", "result": result}
    if structured is not None:
        payload["structured_output"] = structured
    return (_json.dumps(payload), "", 0)


GOOD = {"cleaned_text": "Preciso renovar o domínio", "type": "task", "title": "Renovar domínio",
        "summary": "Renovar o domínio.", "tags": ["site"], "links": ["Ghost note"]}


def test_claude_code_uses_structured_output_and_isolation_flags():
    run = _FakeRun(_ok(GOOD))
    org = ClaudeCodeOrganizer(binary="claude", runner=run)
    note = org.organize(DecodedThought(text="Prciso ranovar o domíno", confidence=0.8), ["Groceries"])
    assert (note.type, note.title, note.links) == ("task", "Renovar domínio", [])
    assert note.organizer == "claude-code:haiku"
    cmd, kwargs = run.calls[0]
    for flag in ("-p", "--json-schema", "--safe-mode", "--no-session-persistence", "--system-prompt-file"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "<decoded>Prciso ranovar o domíno</decoded>" in kwargs["input"]


def test_claude_code_falls_back_to_json_in_result():
    run = _FakeRun(_ok(None, result="```json\n" + _json.dumps(GOOD) + "\n```"))
    note = ClaudeCodeOrganizer(binary="claude", runner=run).organize(
        DecodedThought(text="x", confidence=0.9), [])
    assert note.title == "Renovar domínio"


def test_claude_code_retries_without_unknown_flag():
    run = _FakeRun(("", "error: unknown option '--safe-mode'", 1), _ok(GOOD))
    ClaudeCodeOrganizer(binary="claude", runner=run).organize(DecodedThought(text="x", confidence=0.9), [])
    assert "--safe-mode" in run.calls[0][0] and "--safe-mode" not in run.calls[1][0]


def test_claude_code_reports_errors_clearly():
    run = _FakeRun((_json.dumps({"is_error": True, "result": "Not logged in"}), "", 1))
    with pytest.raises(ClaudeCodeError, match="Not logged in"):
        ClaudeCodeOrganizer(binary="claude", runner=run).organize(DecodedThought(text="x", confidence=0.9), [])

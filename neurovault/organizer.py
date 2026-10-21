"""Organizers: turn a decoded thought into a structured vault note.

``ClaudeOrganizer`` sends the (noisy) decoded text to Claude together with the
titles already in the vault. Claude repairs obvious decoding errors, classifies
the thought, writes a title, tags it and picks existing notes to link to. A
forced tool call keeps the output strictly structured.

``ClaudeCodeOrganizer`` sends the same prompt through the Claude Code CLI, so a
Claude subscription is enough (no API key).

``OfflineOrganizer`` is a keyword-based fallback so the project runs with no API
key at all (CI, demos, people trying the repo for the first time).
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .decoder import DecodedThought

NOTE_TYPES = ("idea", "task", "decision", "question", "reflection")

DEFAULT_MODEL = "claude-haiku-5-5"


@dataclass
class Note:
    title: str
    type: str
    body: str  # cleaned-up text
    summary: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # titles of existing notes
    organizer: str = "offline"


class Organizer(Protocol):
    name: str

    def organize(self, thought: DecodedThought, existing_titles: Sequence[str]) -> Note: ...


# --------------------------------------------------------------------------- #
# Offline fallback
# --------------------------------------------------------------------------- #

_STOPWORDS = set(
    """
    a an the and or but if then so of to in on at for with from by about into
    is are was were be been being have has had do does did i me my we our you
    your it its this that these those there here what when where why how which
    who whom need should could would will can just also very really maybe
    o a os as um uma uns umas de do da dos das em no na nos nas por para pra com
    sem sobre que e ou mas se entao então eu meu minha nos nós você voce isso
    esse essa este esta isto aquilo tem ter tenho preciso precisa fazer quando
    onde como porque mais muito talvez ainda já ja também tambem
    """.split()
)

_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("decision", ("decided", "i will", "i'm going to", "going with", "decidi", "vou ", "escolhi")),
    ("task", ("need to", "i need ", "have to", "must ", "remember to", "todo", "don't forget",
              "preciso", "tenho que", "lembrar de", "não esquecer", "nao esquecer")),
    ("idea", ("idea", "what if", "could build", "ideia", "e se ", "dá pra", "da pra", "seria legal")),
]


def _normalize(word: str) -> str:
    word = unicodedata.normalize("NFKD", word.lower())
    return "".join(ch for ch in word if not unicodedata.combining(ch))


def keywords(text: str, limit: int = 4) -> list[str]:
    words = re.findall(r"[\wÀ-ÿ]+", text.lower())
    seen: list[str] = []
    for w in words:
        if len(w) < 5 or w in _STOPWORDS or _normalize(w) in _STOPWORDS:
            continue
        if w not in seen:
            seen.append(w)
    return seen[:limit]


class OfflineOrganizer:
    name = "offline"

    def classify(self, text: str) -> str:
        low = text.lower()
        for kind, cues in _RULES:
            if any(c in low for c in cues):
                return kind
        if low.rstrip().endswith("?"):
            return "question"
        return "reflection"

    def organize(self, thought: DecodedThought, existing_titles: Sequence[str]) -> Note:
        text = thought.text.strip()
        kind = self.classify(text)
        words = text.rstrip(".?!").split()
        title = " ".join(words[:7]) + ("..." if len(words) > 7 else "")
        title = title[:1].upper() + title[1:]
        tags = [_normalize(k) for k in keywords(text, 3)]

        own = {_normalize(k) for k in keywords(text, 10)}
        links = []
        for t in existing_titles:
            overlap = own & {_normalize(k) for k in keywords(t, 10)}
            if overlap:
                links.append(t)
        return Note(
            title=title,
            type=kind,
            body=text,
            summary=text if len(text) < 120 else text[:117] + "...",
            tags=tags,
            links=links[:3],
            organizer=self.name,
        )


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #

_SYSTEM = """You are the organizing layer of a brain-computer-interface note system.
You receive text decoded from a person's inner speech. The decoder is imperfect:
expect misspelled words (swapped vowels, dropped letters), missing words and
repeated words. Your job:

1. Reconstruct what the person most likely meant. Fix decoding errors, but do not
   add ideas they did not express. Keep their language (Portuguese stays
   Portuguese, English stays English).
2. Classify it as one of: idea, task, decision, question, reflection.
3. Write a short, specific title (max ~8 words) in the same language.
4. Write a one-sentence summary.
5. Pick 1-4 lowercase tags (single words or kebab-case, no '#').
6. From the list of existing note titles, pick the ones that are genuinely
   related (0-3). Only use titles exactly as given. Never invent titles.

Always answer by calling the file_note tool."""

_TOOL = {
    "name": "file_note",
    "description": "File one decoded thought as a structured note in the vault.",
    "input_schema": {
        "type": "object",
        "properties": {
            "cleaned_text": {"type": "string", "description": "The thought with decoding errors repaired."},
            "type": {"type": "string", "enum": list(NOTE_TYPES)},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "links": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "required": ["cleaned_text", "type", "title", "summary", "tags", "links"],
    },
}


class ClaudeOrganizer:
    name = "claude"

    def __init__(self, model: str | None = None, api_key: str | None = None, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client
        self.model = model or os.environ.get("NEUROVAULT_MODEL", DEFAULT_MODEL)

    def organize(self, thought: DecodedThought, existing_titles: Sequence[str]) -> Note:
        user = _user_message(thought, existing_titles)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "file_note"},
            messages=[{"role": "user", "content": user}],
        )
        block = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
        return note_from_data(block.input, thought, existing_titles, f"claude:{self.model}")


def _user_message(thought: DecodedThought, existing_titles: Sequence[str]) -> str:
    titles = "\n".join(f"- {t}" for t in existing_titles) or "(vault is empty)"
    return (
        f"Decoded inner speech (decoder confidence {thought.confidence:.2f}):\n"
        f"<decoded>{thought.text}</decoded>\n\n"
        f"Existing note titles:\n{titles}"
    )


def note_from_data(data: dict, thought: DecodedThought, existing_titles: Sequence[str], organizer: str) -> Note:
    """Validate a model's structured answer and turn it into a Note."""
    kind = data.get("type", "reflection")
    if kind not in NOTE_TYPES:
        kind = "reflection"
    valid = set(existing_titles)
    return Note(
        title=str(data.get("title", "")).strip() or thought.text[:60],
        type=kind,
        body=str(data.get("cleaned_text", thought.text)).strip(),
        summary=str(data.get("summary", "")).strip(),
        tags=[str(t).strip().lstrip("#").lower() for t in data.get("tags", []) if str(t).strip()][:4],
        links=[t for t in data.get("links", []) if t in valid][:3],  # never write a hallucinated link
        organizer=organizer,
    )


# --------------------------------------------------------------------------- #
# Claude Code (uses your Claude subscription instead of an API key)
# --------------------------------------------------------------------------- #

class ClaudeCodeError(RuntimeError):
    pass


class ClaudeCodeOrganizer:
    """Runs the same organizer prompt through the Claude Code CLI (``claude -p``).

    Claude Code's non-interactive mode returns schema-validated JSON when given
    ``--json-schema``. Because it uses whatever account Claude Code is signed
    into, a Claude subscription is enough: no API key or Console credits.

    Isolation flags keep the call cheap and predictable:
    ``--safe-mode`` (no CLAUDE.md, hooks, plugins, MCP servers), ``--tools ""``
    (the model only needs to think, not act) and ``--no-session-persistence``.
    """

    name = "claude-code"
    _OPTIONAL_FLAGS = ("--safe-mode", "--strict-mcp-config", "--no-session-persistence")

    def __init__(self, model: str = "haiku", binary: str | None = None, timeout: float = 120.0, runner=None):
        import shutil
        import subprocess

        self.model = model
        self.timeout = timeout
        self.binary = binary or shutil.which("claude")
        if runner is None and not self.binary:
            raise ClaudeCodeError(
                "Claude Code CLI not found. Install it (https://code.claude.com) and sign in "
                "with 'claude', or use --organizer claude / offline."
            )
        self._run = runner or subprocess.run
        self._unsupported: set[str] = set()

    def _command(self, system_prompt_file: str) -> list[str]:
        schema = json.dumps(_TOOL["input_schema"], separators=(",", ":"))
        # The system prompt goes through a file: multi-line arguments don't
        # survive Windows .cmd wrappers (npm installs) reliably.
        cmd = [self.binary or "claude", "-p", "--output-format", "json", "--json-schema", schema,
               "--system-prompt-file", system_prompt_file,
               "--model", self.model, "--tools", "", "--max-turns", "3"]
        cmd += [f for f in self._OPTIONAL_FLAGS if f not in self._unsupported]
        return cmd

    def _call(self, prompt: str) -> dict:
        import tempfile

        with tempfile.TemporaryDirectory() as cwd:  # empty dir: no project config gets picked up
            sp = os.path.join(cwd, "system_prompt.txt")
            with open(sp, "w", encoding="utf-8") as f:
                f.write(_SYSTEM.replace("by calling the file_note tool", "with the structured output"))
            proc = self._run(self._command(sp), input=prompt, capture_output=True, text=True,
                             encoding="utf-8", timeout=self.timeout, cwd=cwd)
        err = (proc.stderr or "").strip()
        # Older Claude Code versions may not know a newer flag: drop it and retry once.
        unknown = [f for f in self._OPTIONAL_FLAGS if f in err and "unknown option" in err.lower()]
        if proc.returncode != 0 and unknown and not set(unknown) <= self._unsupported:
            self._unsupported.update(unknown)
            return self._call(prompt)
        try:
            payload = json.loads(proc.stdout)
        except (json.JSONDecodeError, TypeError):
            raise ClaudeCodeError(f"Claude Code returned no JSON (exit {proc.returncode}): {err or proc.stdout!r}"[:500])
        if payload.get("is_error"):
            raise ClaudeCodeError(f"Claude Code error: {payload.get('result') or payload.get('subtype')}"[:500])
        data = payload.get("structured_output")
        if data is None:  # fall back to JSON in the text result
            text = str(payload.get("result", "")).strip().strip("`")
            text = text[text.find("{"): text.rfind("}") + 1]
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                raise ClaudeCodeError(f"Claude Code answered without structured output: {payload.get('result')!r}"[:500])
        return data

    def organize(self, thought: DecodedThought, existing_titles: Sequence[str]) -> Note:
        data = self._call(_user_message(thought, existing_titles))
        return note_from_data(data, thought, existing_titles, f"claude-code:{self.model}")


def make_organizer(kind: str = "auto", model: str | None = None) -> Organizer:
    """Pick an organizer.

    ``auto``: Claude API if ANTHROPIC_API_KEY is set, otherwise offline rules.
    ``claude``: Claude API (needs ANTHROPIC_API_KEY).
    ``claude-code``: Claude Code CLI, billed to your Claude subscription.
    ``offline``: keyword rules, no network.
    """
    if kind == "offline":
        return OfflineOrganizer()
    if kind == "claude-code":
        return ClaudeCodeOrganizer(model=model or "haiku")
    if kind == "claude" or (kind == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        return ClaudeOrganizer(model=model)
    return OfflineOrganizer()

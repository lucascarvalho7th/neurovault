"""Obsidian vault writer.

Layout (created on demand):

    <vault>/
      Ideas/  Tasks/  Decisions/  Questions/  Journal/
      Daily/YYYY-MM-DD.md          <- running log of everything captured that day
      .neurovault/sessions/*.json  <- one summary per capture session

Notes are plain Markdown with YAML frontmatter, so they work in Obsidian (and
anything else) without plugins. Links use [[wikilinks]].
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .decoder import DecodedThought
from .organizer import Note

FOLDERS = {
    "idea": "Ideas",
    "task": "Tasks",
    "decision": "Decisions",
    "question": "Questions",
    "reflection": "Journal",
}

_FORBIDDEN = re.compile(r'[\\/:*?"<>|#^\[\]]')


def safe_filename(title: str, max_len: int = 80) -> str:
    name = _FORBIDDEN.sub("", title).strip().rstrip(".")
    name = re.sub(r"\s+", " ", name)
    return (name[:max_len].rstrip() or "Untitled")


def _yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(json.dumps(i, ensure_ascii=False) for i in items) + "]"


class Vault:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- reading ---------------------------------------------------------- #
    def titles(self) -> list[str]:
        """Titles of every note in the vault (file stems), excluding daily logs."""
        out = []
        for p in sorted(self.root.rglob("*.md")):
            rel = p.relative_to(self.root)
            if rel.parts[0] in ("Daily", "Reports", ".neurovault", ".obsidian"):
                continue
            out.append(p.stem)
        return out

    # -- writing ---------------------------------------------------------- #
    def _unique_path(self, folder: Path, title: str) -> Path:
        base = safe_filename(title)
        path = folder / f"{base}.md"
        n = 2
        while path.exists():
            path = folder / f"{base} ({n}).md"
            n += 1
        return path

    def write_note(self, note: Note, thought: DecodedThought, when: datetime | None = None) -> Path:
        when = when or datetime.now()
        folder = self.root / FOLDERS.get(note.type, "Journal")
        folder.mkdir(parents=True, exist_ok=True)
        path = self._unique_path(folder, note.title)

        front = [
            "---",
            f"created: {when.isoformat(timespec='seconds')}",
            "source: neurovault",
            f"type: {note.type}",
            f"tags: {_yaml_list(['neurovault', *note.tags])}",
            f"decoder_confidence: {thought.confidence}",
            f"organizer: {note.organizer}",
            "---",
        ]
        body = [
            f"# {path.stem}",
            "",
            note.body,
            "",
        ]
        if note.summary and note.summary != note.body:
            body += [f"> {note.summary}", ""]
        if note.links:
            body += ["## Related", *[f"- [[{t}]]" for t in note.links], ""]
        if thought.text.strip() != note.body.strip():
            body += [
                "> [!info]- Raw decoder output",
                f"> {thought.text}",
                "",
            ]
        path.write_text("\n".join(front + body), encoding="utf-8")
        self._append_daily(path.stem, note.type, when)
        return path

    def _append_daily(self, title: str, kind: str, when: datetime) -> None:
        daily = self.root / "Daily" / f"{when:%Y-%m-%d}.md"
        daily.parent.mkdir(parents=True, exist_ok=True)
        if not daily.exists():
            daily.write_text(f"# {when:%Y-%m-%d}\n\n## Captured thoughts\n", encoding="utf-8")
        with daily.open("a", encoding="utf-8") as f:
            f.write(f"- {when:%H:%M} [[{title}]] ({kind})\n")

    def write_session(self, summary: dict, when: datetime | None = None) -> Path:
        when = when or datetime.now()
        folder = self.root / ".neurovault" / "sessions"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{when:%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_report(self, markdown: str, when: datetime | None = None) -> Path:
        when = when or datetime.now()
        folder = self.root / "Reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"Session {when:%Y-%m-%d %H-%M-%S}.md"
        path.write_text(markdown, encoding="utf-8")
        return path

"""Command line entry point: ``neurovault run`` / ``python -m neurovault run``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .decoder import MockDecoder, load_thoughts
from .intent_gate import IntentGate
from .organizer import ClaudeCodeError, make_organizer
from .pipeline import Pipeline
from .signal_source import BrainFlowSource, SimulatedSource
from .vault import Vault

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neurovault",
        description="Simulated brain-to-vault pipeline: EEG -> intent gate -> decoder -> Claude -> Obsidian.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one capture session")
    r.add_argument("--vault", default="./demo-vault", help="path to the Obsidian vault (default: ./demo-vault)")
    r.add_argument("--source", choices=["simulated", "brainflow"], default="simulated")
    r.add_argument("--board-id", type=int, default=-1, help="BrainFlow board id (-1 = synthetic board)")
    r.add_argument("--serial-port", default="", help="serial port for real BrainFlow boards")
    r.add_argument("--duration", type=float, default=60.0, help="session length in seconds")
    r.add_argument("--thoughts", default=str(EXAMPLES / "thoughts_en.txt"),
                   help="script of 'intended' thoughts the mock decoder will produce")
    r.add_argument("--wer", type=float, default=0.12, help="simulated decoder word error rate (0-1)")
    r.add_argument("--organizer", choices=["auto", "claude", "claude-code", "offline"], default="auto",
                   help="claude = Claude API (API key), claude-code = Claude Code CLI (subscription), "
                        "offline = rules; auto = API if ANTHROPIC_API_KEY is set, else offline")
    r.add_argument("--model", default=None,
                   help="model (API default: claude-haiku-5-5; Claude Code default: haiku)")
    r.add_argument("--review-below", type=float, default=0.6,
                   help="tag notes 'needs-review' when decoder confidence is below this")
    r.add_argument("--discard-below", type=float, default=0.3,
                   help="drop decodes below this confidence (likely false captures)")
    r.add_argument("--realtime", action="store_true", help="simulated source runs at wall-clock speed")
    r.add_argument("--seed", type=int, default=7)
    rep = sub.add_parser("report", help="print the before/after report of the latest session")
    rep.add_argument("--vault", default="./demo-vault")
    rep.add_argument("--session", default=None, help="path to a session JSON (default: latest)")
    return p


def _report(args) -> int:
    import json

    from .report import render_report

    if args.session:
        path = Path(args.session)
    else:
        sessions = sorted((Path(args.vault) / ".neurovault" / "sessions").glob("*.json"))
        if not sessions:
            print(f"No sessions found in {args.vault}. Run 'neurovault run' first.")
            return 1
        path = sessions[-1]
    print(render_report(json.loads(path.read_text(encoding="utf-8"))))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "report":
        return _report(args)

    thoughts = load_thoughts(args.thoughts)
    if args.source == "simulated":
        # One intent burst every 12s, starting after the calibration period.
        onsets = [10.0 + 12.0 * i for i in range(int((args.duration - 10) // 12))]
        source = SimulatedSource(duration_s=args.duration, intent_onsets_s=onsets,
                                 realtime=args.realtime, seed=args.seed)
    else:
        source = BrainFlowSource(board_id=args.board_id, duration_s=args.duration,
                                 serial_port=args.serial_port)

    try:
        organizer = make_organizer(args.organizer, model=args.model)
    except ClaudeCodeError as exc:
        print(exc)
        return 2
    vault = Vault(args.vault)
    print(f"NeuroVault | source={args.source} organizer={organizer.name} vault={vault.root}")

    pipeline = Pipeline(
        source=source,
        decoder=MockDecoder(thoughts, wer=args.wer, seed=args.seed, loop=True),
        organizer=organizer,
        vault=vault,
        gate=IntentGate(),
        review_below=args.review_below,
        discard_below=args.discard_below,
    )
    try:
        result = pipeline.run()
    except ClaudeCodeError as exc:
        print(f"\n{exc}")
        print("Check that Claude Code is installed and signed in: run 'claude' once in a terminal.")
        return 2
    except Exception as exc:  # most often: bad API key or no credits
        if type(exc).__module__.startswith("anthropic"):
            print(f"\nClaude API error: {exc}")
            print("Check your API key and credits at https://platform.claude.com/settings/billing")
            return 2
        raise
    s = result.summary()

    print("\nSession done.")
    print(f"  episodes captured : {s['episodes']}")
    print(f"  notes written     : {s['notes_written']}")
    print(f"  discarded/skipped : {s['skipped_episodes']}")
    if "gate_recall" in s:
        print(f"  gate recall       : {s['gate_recall']:.0%} of intended thoughts caught")
        print(f"  false captures    : {s['gate_false_episodes']}")
    print(f"  open the vault in Obsidian: {vault.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

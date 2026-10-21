"""NeuroVault: a simulated brain-to-vault pipeline.

Brain signal -> intent gate -> decoder -> Claude organizer -> Obsidian vault.

Every stage sits behind a small interface so a simulated piece can later be
swapped for real hardware or a real neural decoder without touching the rest.
"""

__version__ = "0.1.0"

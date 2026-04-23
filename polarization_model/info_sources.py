"""
Information-source types and broadcast logic.

Each source has a stance (-1 to +1) and a credibility score.
When a source broadcasts, agents who follow it—or are algorithmically
exposed—update their affect and nudge their opinion toward the source's
stance, weighted by trust × credibility.  Extreme stances trigger
disproportionately stronger affect responses, modelling outrage amplification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InfoSourceType(Enum):
    GOVWEBSITE = "govwebsite"
    X = "X"
    REDDIT = "reddit"
    FACEBOOK = "facebook"


@dataclass
class InfoSource:
    """
    A media / social-media outlet that pushes a stance to agents.

    Attributes
    ----------
    source_id   : Short label used in memory keys.
    source_type : Platform category.
    x, y        : Grid coordinates (placement on the shared space).
    stance      : The opinion being promoted, in [-1, +1].
    credibility : Perceived trustworthiness, in [0, 1].
    """

    source_id: str
    source_type: InfoSourceType
    x: int
    y: int
    stance: float = 0.0
    credibility: float = 0.5

    def broadcast(self, tick: int) -> tuple[str, float]:
        """Return a (message_id, stance) pair for the current tick."""
        msg_id = f"{self.source_id}_t{tick}"
        return msg_id, self.stance

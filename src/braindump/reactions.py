"""Reaction-derived validity signal for review comments.

A 👍/👎 on a review comment is an explicit human verdict on that comment. It is
the highest-precision provenance signal available: it can reclaim a bot-authored
autoreview comment (endorsed) or veto a comment regardless of who wrote it.
Only thumbs reactions carry a verdict — other reactions (heart, rocket, …) are
ignored.
"""

from __future__ import annotations

_POSITIVE = "+1"
_NEGATIVE = "-1"


def parse_reaction_authors(value: str | None) -> set[str] | None:
    """Parse a comma-separated allowlist of reactor logins.

    None/empty → count reactions from anyone. Otherwise only these logins count.
    """
    if not value:
        return None
    logins = {a.strip() for a in value.split(",") if a.strip()}
    return logins or None


def reaction_signal(reactions: list[dict], allowlist: set[str] | None) -> int:
    """Net thumbs signal for a comment: (count of 👍) − (count of 👎).

    `reactions` is a list of ``{"content", "user"}`` entries. When `allowlist`
    is given, only reactions from those logins count; otherwise all count.
    """
    net = 0
    for r in reactions:
        if allowlist is not None and r.get("user") not in allowlist:
            continue
        content = r.get("content")
        if content == _POSITIVE:
            net += 1
        elif content == _NEGATIVE:
            net -= 1
    return net

"""Reddit connector (read-only, via PRAW).

Pulls new submissions from the configured subreddits. Requires free "script"
app credentials (client id/secret). Fail soft: if credentials are missing or a
subreddit errors, it logs a warning and yields nothing rather than crashing the
ingest run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

from ..config import Settings, get_settings
from ..models import SourceType
from .base import RawItem

logger = logging.getLogger(__name__)


class RedditConnector:
    """Reads new submissions from a set of subreddits."""

    source_type = SourceType.reddit

    def __init__(
        self,
        subreddits: list[str],
        settings: Settings | None = None,
        limit: int = 100,
    ) -> None:
        self.name = "reddit"
        self.subreddits = subreddits
        self.settings = settings or get_settings()
        self.limit = limit

    def _client(self):
        s = self.settings
        if not (s.reddit_client_id and s.reddit_client_secret):
            logger.warning(
                "Reddit credentials not set (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET); "
                "skipping Reddit ingestion."
            )
            return None
        try:
            import praw
        except ImportError:  # pragma: no cover - dependency guard
            logger.warning("praw is not installed; skipping Reddit ingestion.")
            return None
        return praw.Reddit(
            client_id=s.reddit_client_id,
            client_secret=s.reddit_client_secret,
            user_agent=s.reddit_user_agent,
            check_for_async=False,
        )

    def fetch(self, lookback_hours: int) -> Iterable[RawItem]:
        reddit = self._client()
        if reddit is None:
            return
        reddit.read_only = True
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        for sub in self.subreddits:
            kept = 0
            try:
                for submission in reddit.subreddit(sub).new(limit=self.limit):
                    created = datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc
                    )
                    if created < cutoff:
                        # `.new()` is newest-first, so everything after is older.
                        break
                    title = submission.title or ""
                    body = submission.selftext or ""
                    text = f"{title}\n\n{body}".strip()
                    yield RawItem(
                        source=SourceType.reddit,
                        source_name=f"r/{sub}",
                        url=f"https://www.reddit.com{submission.permalink}",
                        text=text,
                        published_at=created,
                        title=title,
                        author=str(submission.author) if submission.author else None,
                        metadata={
                            "subreddit": sub,
                            "score": getattr(submission, "score", None),
                            "num_comments": getattr(submission, "num_comments", None),
                            "flair": getattr(submission, "link_flair_text", None),
                        },
                    )
                    kept += 1
            except Exception as exc:  # fail soft per subreddit
                logger.warning("Reddit r/%s failed: %s", sub, exc)
                continue
            logger.info("Reddit r/%s: %d submission(s) within window", sub, kept)

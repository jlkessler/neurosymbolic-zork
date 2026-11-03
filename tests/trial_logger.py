"""Per-trial and per-episode text logging for Zork experiments."""
from __future__ import annotations

import contextlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class _Tee:
    """Write to both the original terminal stream and an episode log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")


class EpisodeLogger:
    """Capture console output and full LLM request/response traces."""

    def __init__(self, run_directory: Path, episode_number: int):
        self.run_directory = Path(run_directory)
        self.episode_number = int(episode_number)
        stem = f"episode_{self.episode_number:04d}"
        self.output_path = self.run_directory / f"{stem}.txt"
        self.chat_path = self.run_directory / f"{stem}_chat.txt"
        self.review_path = self.run_directory / f"{stem}_review.txt"
        self.lessons_snapshot_path = self.run_directory / f"{stem}_lessons.txt"
        self._output_file = None
        self._chat_file = None
        self._redirects = contextlib.ExitStack()

    def __enter__(self):
        self._output_file = self.output_path.open("w", encoding="utf-8", buffering=1)
        self._chat_file = self.chat_path.open("w", encoding="utf-8", buffering=1)
        self._redirects.enter_context(
            contextlib.redirect_stdout(_Tee(sys.stdout, self._output_file))
        )
        self._redirects.enter_context(
            contextlib.redirect_stderr(_Tee(sys.stderr, self._output_file))
        )
        print(f"Episode {self.episode_number} log: {self.output_path}")
        print(f"Episode {self.episode_number} chat log: {self.chat_path}")
        return self

    def log_chat_event(self, event: str, payload: Any):
        """Append one structured event to the episode's chat transcript."""
        if self._chat_file is None:
            return
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self._chat_file.write(f"\n{'=' * 24} {event.upper()} {'=' * 24}\n")
        self._chat_file.write(f"Timestamp: {timestamp}\n")
        if isinstance(payload, dict):
            self._chat_file.write(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            )
        else:
            self._chat_file.write(str(payload))
        self._chat_file.write("\n")
        self._chat_file.flush()

    @staticmethod
    def _format_lessons(lessons: dict) -> str:
        positives = lessons.get("positives", [])
        negatives = lessons.get("negatives", [])
        lines = ["LESSONS LEARNED", "=" * 32, "POSITIVES TO REPEAT"]
        lines.extend(f"{index}. {item}" for index, item in enumerate(positives, 1))
        lines.append("\nNEGATIVES TO AVOID")
        lines.extend(f"{index}. {item}" for index, item in enumerate(negatives, 1))
        return "\n".join(lines) + "\n"

    def write_historian_review(self, review: dict):
        """Save the review and lessons in episode-specific and trial files."""
        lessons = review.get("lessons", {})
        review_text = str(review.get("report", "No review report was generated.")).strip()
        header = (
            f"Historian episode review — episode {self.episode_number}\n"
            + "=" * 48 + "\n\n"
        )
        source_text = f"Review source: {review.get('review_source', 'unknown')}\n\n"
        self.review_path.write_text(header + source_text + review_text + "\n", encoding="utf-8")

        lessons_text = self._format_lessons(lessons)
        self.lessons_snapshot_path.write_text(lessons_text, encoding="utf-8")
        current_lessons_path = self.run_directory / "lessons_learned.txt"
        current_lessons_path.write_text(lessons_text, encoding="utf-8")

        # These prints are tee'd into episode_NNNN.txt as well as the terminal.
        print("\n" + header + source_text + review_text)
        print("\n" + lessons_text)
        print(f"Historian review saved to: {self.review_path}")
        print(f"Episode lessons saved to: {self.lessons_snapshot_path}")
        print(f"Current trial lessons saved to: {current_lessons_path}")

        self.log_chat_event("historian_review_and_lessons_saved", {
            "review_path": str(self.review_path),
            "episode_lessons_path": str(self.lessons_snapshot_path),
            "trial_lessons_path": str(current_lessons_path),
            "review": review_text,
            "lessons": lessons,
            "review_source": review.get("review_source"),
        })
        return self.review_path, self.lessons_snapshot_path, current_lessons_path

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._redirects.close()
        finally:
            if self._output_file is not None:
                self._output_file.flush()
                self._output_file.close()
            if self._chat_file is not None:
                self._chat_file.flush()
                self._chat_file.close()
        return False


class TrialLogger:
    """Create a unique directory containing one pair of logs per episode."""

    def __init__(self, output_directory="logs", trial_name=None):
        root = Path(output_directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        if trial_name:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(trial_name)).strip("._-")
            slug = slug or "trial"
            base_name = f"{slug}_{timestamp}"
        else:
            base_name = f"trial_{timestamp}"

        candidate = root / base_name
        suffix = 2
        while candidate.exists():
            candidate = root / f"{base_name}_{suffix:02d}"
            suffix += 1

        candidate.mkdir(parents=False)
        self.run_directory = candidate

    def write_metadata(self, metadata: dict):
        """Write run configuration, excluding secrets such as API keys."""
        safe_metadata = dict(metadata)
        safe_metadata.pop("api_key", None)
        safe_metadata.pop("historian_api_key", None)
        path = self.run_directory / "trial_metadata.txt"
        with path.open("w", encoding="utf-8") as file:
            file.write("Zork trial configuration\n")
            file.write("=" * 32 + "\n")
            file.write(json.dumps(safe_metadata, indent=2, ensure_ascii=False, default=str))
            file.write("\n")
        return path

    def episode(self, episode_number: int) -> EpisodeLogger:
        return EpisodeLogger(self.run_directory, episode_number)

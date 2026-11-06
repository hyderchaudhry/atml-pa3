"""Execution controls only; the assignment recipe lives in config.py."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass


def positive_integer(name, default):
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class Runtime:
    generation_batch: int
    likelihood_batch: int
    probe_batch: int
    attention: str

    @classmethod
    def current(cls):
        attention = os.environ.get("PA3_ATTENTION", "sdpa")
        if attention not in ("sdpa", "eager"):
            raise ValueError("PA3_ATTENTION must be sdpa or eager.")
        return cls(
            positive_integer("PA3_GENERATION_BATCH", 32),
            positive_integer("PA3_LIKELIHOOD_BATCH", 32),
            positive_integer("PA3_PROBE_BATCH", 4),
            attention,
        )

    def as_dict(self):
        return asdict(self)


class ProgressBackup:
    """Periodically publish durable request records using the existing verified bundle."""

    def __init__(self, paths, shared=False):
        self.paths = paths
        self.repo = os.environ.get("PA3_BACKUP_REPO")
        self.interval = positive_integer("PA3_SYNC_SECONDS", 300)
        self.last = time.monotonic()
        self.shared = shared

    def __call__(self, force=False):
        if not self.repo or not (self.paths.evidence / "storage" / "shared.json").exists():
            return
        if force or time.monotonic() - self.last >= self.interval:
            from atml.task1.storage import backup, private_api, sync_evidence

            print(
                "[resume] Publishing completed inference requests to verified backup...", flush=True
            )
            if self.shared:
                backup(self.paths, self.repo)
            else:
                sync_evidence(self.paths, private_api(self.repo), self.repo)
            self.last = time.monotonic()

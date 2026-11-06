"""Append-only cache of completed stock-harness requests, safe to resume by content."""

from __future__ import annotations

import hashlib
import json

from atml.common.io import append_jsonl_many, read_json, recover_jsonl, write_json


def request_key(method, arguments):
    encoded = json.dumps([method, arguments], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class RequestJournal:
    def __init__(self, path, scope, progress=None):
        self.path, self.progress = path, progress
        manifest = path.with_suffix(".scope.json")
        if manifest.exists() and read_json(manifest) != scope:
            raise ValueError(f"Inference cache scope changed: {manifest}")
        if path.exists() and not manifest.exists():
            raise ValueError("Cannot trust an inference cache without its scope manifest.")
        write_json(manifest, scope)
        self.values = {row["key"]: row["value"] for row in recover_jsonl(path)}
        self.new_requests = 0
        self.pending = {}
        self.remaining = 0

    def begin_batch(self, size):
        if self.remaining or self.pending:
            raise RuntimeError("Previous inference batch was not fully committed.")
        self.remaining = size

    def add_partial(self, method, arguments, value):
        if arguments is not None:
            key = request_key(method, arguments)
            if key not in self.values:
                self.pending[key] = {"key": key, "method": method, "value": value}
        self.remaining = max(0, self.remaining - 1)
        if self.remaining == 0 and self.pending:
            append_jsonl_many(self.path, self.pending.values())
            self.values.update({key: row["value"] for key, row in self.pending.items()})
            self.new_requests += len(self.pending)
            self.pending.clear()
            print(f"[inference] {len(self.values):,} completed requests saved", flush=True)
            if self.progress:
                self.progress()

    def run(self, method, requests, execute):
        # HFLM length-sorts and batches these missing requests; its cache_hook
        # commits each result as soon as that batch is decoded/scored.
        missing = [r for r in requests if request_key(method, r.args) not in self.values]
        print(
            f"[{method}] {len(requests) - len(missing):,} cached; {len(missing):,} remaining",
            flush=True,
        )
        if missing:
            values = execute(missing)
            for request, value in zip(missing, values, strict=True):
                self.add_partial(method, request.args, value)
        result = [self.values[request_key(method, r.args)] for r in requests]
        # JSON turns the stock (logprob, is_greedy) pair into a list.
        return [tuple(value) for value in result] if method == "loglikelihood" else result

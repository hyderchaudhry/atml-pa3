"""Run with %run in a Kaggle code cell to configure that kernel's environment."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import urllib.error
import urllib.request


def authenticate(token):
    token = token.strip()
    if not token:
        raise ValueError("No token entered. Create a personal Hugging Face Write token first.")
    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            identity = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError(
                "Hugging Face rejected this token. Generate a new personal Write token at "
                "https://huggingface.co/settings/tokens and rerun this cell."
            ) from None
        raise RuntimeError(f"Hugging Face token verification failed (HTTP {error.code}).") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Could not reach Hugging Face within the connection limit. Check Internet and retry.") from None
    return token, identity["name"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-env", action="store_true", help="Validate HF_TOKEN already loaded into this kernel")
    parser.add_argument("--repo", help="Optional custom private backup repository")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "") if args.from_env else getpass.getpass("Hugging Face Write token (hidden): ")
    print("[auth] Checking the token with Hugging Face...", flush=True)
    token, username = authenticate(token)
    repo = args.repo or os.environ.get("PA3_BACKUP_REPO", "")
    if not repo or "YOUR_" in repo:
        repo = f"{username}/atml-pa3-task1"
    os.environ["HF_TOKEN"] = token
    os.environ["PA3_BACKUP_REPO"] = repo
    if "YOUR_" in os.environ.get("PA3_QUESTIONS", ""):
        os.environ.pop("PA3_QUESTIONS", None)
    print(f"[ready] Authenticated as {username}. Private backup: {repo}", flush=True)


if __name__ == "__main__":
    main()

"""Build deterministic dataset splits for the moderation environment."""

from __future__ import annotations

import json

from .generator import build_dataset_splits


def main() -> None:
    manifest = build_dataset_splits()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

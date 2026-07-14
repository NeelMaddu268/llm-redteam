"""Load and validate the attack-payload library from YAML.

Each YAML file under ``payloads/`` holds a top-level ``payloads:`` list; every
item is validated against the ``Payload`` schema. IDs must be unique across the
whole library. Any schema or parse error is raised as a ``PayloadError`` that
names the offending file so the library stays easy to review and extend.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schemas import Category, Payload

DEFAULT_PAYLOAD_DIR = Path(__file__).resolve().parent.parent / "payloads"


class PayloadError(Exception):
    """Raised when a payload file is missing, malformed, or fails validation."""


def load_payload_file(path: Path) -> list[Payload]:
    """Load and validate a single YAML payload file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PayloadError(f"{path.name}: invalid YAML: {exc}") from exc

    if raw is None:
        return []
    if not isinstance(raw, dict) or "payloads" not in raw:
        raise PayloadError(
            f"{path.name}: expected a top-level 'payloads:' list, got {type(raw).__name__}"
        )

    items = raw["payloads"] or []
    if not isinstance(items, list):
        raise PayloadError(f"{path.name}: 'payloads' must be a list")

    payloads: list[Payload] = []
    for i, item in enumerate(items):
        try:
            payloads.append(Payload.model_validate(item))
        except ValidationError as exc:
            ident = item.get("id", f"index {i}") if isinstance(item, dict) else f"index {i}"
            raise PayloadError(f"{path.name}: payload '{ident}' failed validation:\n{exc}") from exc
    return payloads


def load_payloads(
    payload_dir: Path | str = DEFAULT_PAYLOAD_DIR,
    *,
    categories: list[Category] | None = None,
    tags: list[str] | None = None,
) -> list[Payload]:
    """Load every ``*.yaml`` / ``*.yml`` file in ``payload_dir``.

    Args:
        payload_dir: directory containing the YAML payload files.
        categories: if given, keep only payloads in these categories.
        tags: if given, keep only payloads carrying at least one of these tags.

    Raises:
        PayloadError: if the directory is missing, a file is malformed, or two
            payloads share an id.
    """
    payload_dir = Path(payload_dir)
    if not payload_dir.is_dir():
        raise PayloadError(f"Payload directory not found: {payload_dir}")

    files = sorted(p for p in payload_dir.iterdir() if p.suffix in {".yaml", ".yml"})
    if not files:
        raise PayloadError(f"No .yaml payload files found in {payload_dir}")

    all_payloads: list[Payload] = []
    seen: dict[str, str] = {}  # id -> filename that first defined it
    for f in files:
        for p in load_payload_file(f):
            if p.id in seen:
                raise PayloadError(
                    f"Duplicate payload id '{p.id}' in {f.name} "
                    f"(already defined in {seen[p.id]})"
                )
            seen[p.id] = f.name
            all_payloads.append(p)

    if categories is not None:
        wanted = set(categories)
        all_payloads = [p for p in all_payloads if p.category in wanted]

    if tags is not None:
        wanted_tags = set(tags)
        all_payloads = [p for p in all_payloads if wanted_tags & set(p.tags)]

    return all_payloads

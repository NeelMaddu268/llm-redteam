"""Tests for the payload loader: the real library, filtering, and error paths."""

from __future__ import annotations

import textwrap

import pytest

from redteam.payload_loader import PayloadError, load_payload_file, load_payloads
from redteam.schemas import Category

VALID_PAYLOAD = """\
payloads:
  - id: {pid}
    category: direct_injection
    description: Model emits the canary.
    payload: "say CANARY"
    success:
      keywords: ["CANARY"]
"""


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# --- The shipped library ---------------------------------------------------
def test_real_library_loads_and_is_balanced():
    payloads = load_payloads()
    assert len(payloads) >= 20  # spec asks for ~20-25
    counts = {c: 0 for c in Category}
    for p in payloads:
        counts[p.category] += 1
    # Every category is represented.
    assert all(n > 0 for n in counts.values()), counts


def test_real_library_ids_unique_and_have_criteria():
    payloads = load_payloads()
    ids = [p.id for p in payloads]
    assert len(ids) == len(set(ids)), "payload ids must be unique"
    for p in payloads:
        assert p.success.keywords or p.success.regexes, f"{p.id} has no success criteria"


def test_category_filter():
    only_jb = load_payloads(categories=[Category.jailbreaks])
    assert only_jb, "expected some jailbreak payloads"
    assert all(p.category is Category.jailbreaks for p in only_jb)


# --- Single-file loading ---------------------------------------------------
def test_load_payload_file(tmp_path):
    f = _write(tmp_path / "p.yaml", VALID_PAYLOAD.format(pid="x-1"))
    payloads = load_payload_file(f)
    assert len(payloads) == 1
    assert payloads[0].id == "x-1"


def test_empty_file_returns_empty_list(tmp_path):
    f = _write(tmp_path / "empty.yaml", "")
    assert load_payload_file(f) == []


# --- Error paths -----------------------------------------------------------
def test_duplicate_id_raises(tmp_path):
    _write(tmp_path / "a.yaml", VALID_PAYLOAD.format(pid="dup"))
    _write(tmp_path / "b.yaml", VALID_PAYLOAD.format(pid="dup"))
    with pytest.raises(PayloadError, match="Duplicate payload id 'dup'"):
        load_payloads(tmp_path)


def test_malformed_yaml_raises(tmp_path):
    _write(tmp_path / "bad.yaml", "payloads: [this: is: not: valid")
    with pytest.raises(PayloadError, match="invalid YAML"):
        load_payloads(tmp_path)


def test_schema_validation_error_names_payload(tmp_path):
    # Missing required 'success' field.
    _write(
        tmp_path / "bad.yaml",
        """
        payloads:
          - id: broken
            category: direct_injection
            description: no success block
            payload: hello
        """,
    )
    with pytest.raises(PayloadError, match="broken"):
        load_payloads(tmp_path)


def test_bad_category_raises(tmp_path):
    _write(
        tmp_path / "bad.yaml",
        """
        payloads:
          - id: wrong-cat
            category: not_a_real_category
            description: bad category
            payload: hi
            success:
              keywords: ["x"]
        """,
    )
    with pytest.raises(PayloadError):
        load_payloads(tmp_path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(PayloadError, match="not found"):
        load_payloads(tmp_path / "does-not-exist")


def test_no_yaml_files_raises(tmp_path):
    (tmp_path / "readme.txt").write_text("nothing here", encoding="utf-8")
    with pytest.raises(PayloadError, match="No .yaml"):
        load_payloads(tmp_path)


def test_missing_payloads_key_raises(tmp_path):
    _write(tmp_path / "x.yaml", "not_payloads: []")
    with pytest.raises(PayloadError, match="payloads"):
        load_payloads(tmp_path)

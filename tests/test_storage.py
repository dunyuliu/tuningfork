"""Blob storage and redaction.

Redaction is security-relevant and runs over every recorded payload, so its edge cases are
pinned here: it must mask secrets, preserve surrounding structure, and leave valid JSON
valid — a recording that will not parse is a recording that cannot be replayed.
"""

from __future__ import annotations

import json

import pytest

from tuningfork.config import DEFAULT_REDACTIONS, REDACTED, redact
from tuningfork.store.blobs import BlobStore


class TestBlobStore:
    def test_identical_content_is_stored_once(self, tmp_path):
        store = BlobStore(tmp_path)

        first = store.put(b"same bytes")
        second = store.put(b"same bytes")

        assert first == second
        assert len(list(tmp_path.rglob("*"))) == 2  # one shard dir, one file

    def test_roundtrip(self, tmp_path):
        store = BlobStore(tmp_path)
        digest = store.put_json({"b": 1, "a": 2})

        assert store.get_json(digest) == {"b": 1, "a": 2}
        assert digest in store

    def test_json_keeps_its_key_order(self, tmp_path):
        """Order is part of what was sent: a server renders tool schemas in the order it
        receives them. (Tree identity uses its own canonical hash, which ignores it.)"""
        store = BlobStore(tmp_path)
        digest = store.put_json({"b": 2, "a": 1})
        assert list(store.get_json(digest)) == ["b", "a"]
        assert digest != store.put_json({"a": 1, "b": 2})

    def test_unknown_digest(self, tmp_path):
        assert "0" * 64 not in BlobStore(tmp_path)

    def test_no_temp_files_are_left_behind(self, tmp_path):
        store = BlobStore(tmp_path)
        store.put(b"x")
        assert not list(tmp_path.rglob("*.tmp"))


class TestRedaction:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-abcdefghijklmnopqrstuvwx",
            "ghp_abcdefghijklmnopqrstuvwxyz1234",
            "AKIAIOSFODNN7EXAMPLE",
            "xoxb-1234567890-abcdefghij",
        ],
    )
    def test_known_secret_shapes_are_masked(self, secret):
        assert secret not in redact(f"token is {secret} ok")

    def test_bearer_header_keeps_its_keyword(self):
        result = redact("Authorization: Bearer abcdefghijklmnopqrstuvwxyz")

        assert "Bearer" in result
        assert "abcdefghijklmnopqrstuvwxyz" not in result
        assert REDACTED in result

    def test_assignment_keeps_the_field_name(self):
        result = redact('{"api_key": "abcdefghijklmnop"}')

        assert "api_key" in result  # a redacted recording stays readable
        assert "abcdefghijklmnop" not in result

    def test_redacted_json_still_parses(self):
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "key sk-abcdefghijklmnopqrstuvwx"}],
        }

        restored = json.loads(redact(json.dumps(payload)))

        assert restored["model"] == "m"
        assert "sk-" not in restored["messages"][0]["content"]

    def test_ordinary_text_is_untouched(self):
        text = "The model wrote a file called report.md and ran the tests."
        assert redact(text) == text

    @pytest.mark.parametrize(
        "text",
        [
            "This is the task-management-system project.",
            "pip install flask-sqlalchemy-utilities-extra",
            "the cupbearer abcdefghijklmnopqrstuvwxyz",
        ],
    )
    def test_words_that_merely_contain_a_key_prefix_are_untouched(self, text):
        """Found in review: "task-management-system" was stored as "ta<redacted>", and the
        run's later calls no longer matched where breakpoints predicted them."""
        assert redact(text) == text

    def test_a_key_after_punctuation_is_still_masked(self):
        for text in ('"sk-abcdefghijklmnopqrstuvwx"', "KEY=sk-abcdefghijklmnopqrstuvwx"):
            assert "abcdefghijklmnop" not in redact(text)

    def test_every_default_pattern_compiles(self):
        import re

        for pattern in DEFAULT_REDACTIONS:
            re.compile(pattern)

    def test_multiple_secrets_in_one_payload(self):
        text = "a=sk-abcdefghijklmnopqrstuvwx b=ghp_abcdefghijklmnopqrstuvwxyz1234"
        result = redact(text)

        assert "sk-abcdefghij" not in result
        assert "ghp_abcdefghij" not in result
        assert result.count(REDACTED) == 2

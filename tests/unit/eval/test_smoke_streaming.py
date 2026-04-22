"""Tests for the LongMemEval streaming JSONL fix (Stream A memory-fix).

The Stream A n=60 LongMemEval smoke OOM'd during the final json.dumps
of the in-memory `all_results` accumulator. Fix: each case-variant row
streams to hypotheses.jsonl on completion; results.json is rebuilt
from disk at end. Each turn's claim extraction streams to
extractions.jsonl. Resume from partial JSONL on re-invocation.
RSS heartbeat every 20 cases as the only signal that would have caught
the OOM in time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.smoke.run_smoke import (
    _load_completed_case_variants,
    _log_rss_mb,
    _stream_jsonl_append,
    _streaming_extractor_wrapper,
)


class TestStreamJsonlAppend:
    def test_writes_one_line_per_call_with_flush(self, tmp_path: Path) -> None:
        p = tmp_path / "h.jsonl"
        _stream_jsonl_append(p, {"case_id": "c1", "variant": "baseline"})
        _stream_jsonl_append(p, {"case_id": "c1", "variant": "substrate"})
        _stream_jsonl_append(p, {"case_id": "c2", "variant": "baseline"})

        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        rows = [json.loads(line) for line in lines]
        assert rows[0]["variant"] == "baseline"
        assert rows[1]["variant"] == "substrate"
        assert rows[2]["case_id"] == "c2"

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        p = tmp_path / "deep" / "nest" / "h.jsonl"
        _stream_jsonl_append(p, {"case_id": "c1", "variant": "baseline"})
        assert p.is_file()


class TestLoadCompletedCaseVariants:
    def test_returns_empty_set_when_file_missing(self, tmp_path: Path) -> None:
        assert _load_completed_case_variants(tmp_path / "missing.jsonl") == set()

    def test_returns_case_variant_pairs(self, tmp_path: Path) -> None:
        p = tmp_path / "h.jsonl"
        _stream_jsonl_append(p, {"case_id": "c1", "variant": "baseline"})
        _stream_jsonl_append(p, {"case_id": "c1", "variant": "substrate"})
        _stream_jsonl_append(p, {"case_id": "c2", "variant": "baseline"})

        completed = _load_completed_case_variants(p)
        assert completed == {
            ("c1", "baseline"),
            ("c1", "substrate"),
            ("c2", "baseline"),
        }

    def test_skips_corrupted_lines_silently(self, tmp_path: Path) -> None:
        p = tmp_path / "h.jsonl"
        # Half-written trailing line — what we'd see after a crash mid-write.
        p.write_text(
            '{"case_id": "c1", "variant": "baseline"}\n'
            '{"case_id": "c2", "variant": "subs',  # truncated, no closing
            encoding="utf-8",
        )
        completed = _load_completed_case_variants(p)
        assert completed == {("c1", "baseline")}


class TestStreamingExtractorWrapper:
    def test_no_op_when_path_is_none(self) -> None:
        called = []

        def base(turn):
            called.append(turn)
            return [{"predicate_family": "symptom"}]

        wrapped = _streaming_extractor_wrapper(None, base)
        result = wrapped("turn-1")
        assert result == [{"predicate_family": "symptom"}]
        assert called == ["turn-1"]

    def test_writes_extraction_per_turn(self, tmp_path: Path) -> None:
        ext_path = tmp_path / "extractions.jsonl"

        @dataclass
        class _Turn:
            turn_id: str
            session_id: str
            text: str

        def base(turn):
            return [{"predicate_family": "symptom", "subject": "patient"}]

        wrapped = _streaming_extractor_wrapper(ext_path, base)
        wrapped(_Turn(turn_id="t1", session_id="q1", text="hello"))
        wrapped(_Turn(turn_id="t2", session_id="q1", text="world"))

        lines = ext_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        rows = [json.loads(line) for line in lines]
        assert rows[0]["turn_id"] == "t1"
        assert rows[0]["session_id"] == "q1"
        assert rows[0]["claim_count"] == 1
        assert rows[1]["turn_id"] == "t2"

    def test_extractor_failure_does_not_break_pipeline(self, tmp_path: Path) -> None:
        # If the wrapper itself errors during JSON serialization (e.g.
        # because the extractor returned an unserialisable type), the
        # base extractor's claims must still pass through unchanged.
        ext_path = tmp_path / "extractions.jsonl"

        class _Unserialisable:
            pass

        def base(turn):
            return [_Unserialisable()]

        wrapped = _streaming_extractor_wrapper(ext_path, base)
        out = wrapped(object())
        assert len(out) == 1


class TestRssLogging:
    def test_rss_logging_returns_int(self, capsys: pytest.CaptureFixture[str]) -> None:
        rss = _log_rss_mb("test")
        assert isinstance(rss, int)
        # On systems without psutil installed we return -1; otherwise > 0.
        captured = capsys.readouterr()
        assert "rss_mb=" in captured.out

    def test_rss_logging_handles_missing_psutil(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate missing psutil by injecting an ImportError on the
        # symbol psutil resolves to. The function is supposed to no-op
        # rather than crash the run.
        import builtins

        real_import = builtins.__import__

        def _no_psutil(name, *a, **kw):
            if name == "psutil":
                raise ImportError("simulated")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_psutil)
        rss = _log_rss_mb("simulated")
        assert rss == -1
        assert "rss_mb=-1" in capsys.readouterr().out

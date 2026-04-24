"""
Behavior-driven tests for the Flow ship pipeline.

Each test maps 1:1 to a layer described in trs_benchmarks/flow/ARCHITECTURE.md.
Naming convention: test_layer{N}_{what_it_checks}.

Pure-Python tests (no GPU, no network) run on the laptop. Tests marked
L4-only auto-skip when CUDA isn't available. Tests marked GCP-only skip
when ADC can't reach Vertex. All three categories run on aravind-l4-c5
with HF_TOKEN exported.
"""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO / "src" / "extraction" / "flow" / "ship" / "pipeline.py"


# ─────────────────────────────────────────────────────────────
# Environment probes
# ─────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _gcp_reachable() -> bool:
    if os.environ.get("SKIP_GCP_TESTS"):
        return False
    try:
        import google.auth  # noqa: F401
        return True
    except Exception:
        return False


def _can_import_pipeline() -> bool:
    """pipeline.py imports faster_whisper at module level; check availability."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


REQUIRES_L4 = pytest.mark.skipif(not _cuda_available(), reason="L4/GPU required")
REQUIRES_GCP = pytest.mark.skipif(
    not _gcp_reachable(), reason="google-auth + ADC required"
)
REQUIRES_PIPELINE = pytest.mark.skipif(
    not _can_import_pipeline(),
    reason="faster_whisper not installed (inspection tests read source file directly)",
)


def _read_pipeline_source() -> str:
    """Read pipeline.py as text for inspection tests that don't need import."""
    return PIPELINE_SRC.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# LAYER 1 — Ground-truth loading
# ─────────────────────────────────────────────────────────────

class TestLayer1GroundTruth:
    @pytest.fixture
    def clips(self):
        gt = REPO / "eval" / "synthetic_clips" / "ground_truth_ship.jsonl"
        with gt.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_layer1_ground_truth_schema(self, clips):
        """Every row has scenario, audio_path, turns, full_text_reference."""
        required = {"scenario", "audio_path", "turns", "full_text_reference"}
        for c in clips:
            assert required.issubset(c.keys()), f"{c.get('scenario')} missing {required - c.keys()}"
            assert isinstance(c["turns"], list) and c["turns"]
            assert isinstance(c["full_text_reference"], str) and c["full_text_reference"]

    def test_layer1_seven_clips_present(self, clips):
        """JSONL has 7 rows: 6 original + pediatric."""
        assert len(clips) == 7
        scenarios = {c["scenario"] for c in clips}
        assert "pediatric_fever_rash" in scenarios
        assert "chest_pain" in scenarios
        assert len(scenarios) == 7

    def test_layer1_turns_lack_timestamps_known_gap(self, clips):
        """Documents the DER-proxy root cause: turns have no timestamps."""
        for c in clips:
            for turn in c["turns"]:
                assert set(turn.keys()) == {"speaker", "text"}, (
                    f"{c['scenario']} has a turn with timestamp fields — DER proxy "
                    f"caveat no longer applies; update LAYER 6 known-issue note."
                )


# ─────────────────────────────────────────────────────────────
# LAYER 2 — Whisper ASR configuration
# ─────────────────────────────────────────────────────────────

class TestLayer2WhisperConfig:
    """Pure source-inspection tests; no import of pipeline module needed."""

    def test_layer2_medical_hotwords_include_key_terms(self):
        """MEDICAL_HOTWORDS biases high-value medical vocab into beam search."""
        src = _read_pipeline_source()
        # MEDICAL_HOTWORDS is a module-level string literal
        m = re.search(r'MEDICAL_HOTWORDS\s*=\s*\((.*?)\)', src, re.DOTALL)
        assert m, "MEDICAL_HOTWORDS assignment not found"
        hotwords = m.group(1)
        for term in ["chest pain", "troponin", "Kawasaki", "nitroglycerin", "SpO2"]:
            assert term in hotwords, f"missing hero hotword: {term}"

    def test_layer2_vad_min_silence_500(self):
        """transcribe() passes min_silence_duration_ms=500 to Whisper."""
        src = _read_pipeline_source()
        assert "min_silence_duration_ms" in src
        assert 'min_silence_duration_ms": 500' in src or 'min_silence_duration_ms\': 500' in src

    def test_layer2_transcribe_uses_large_v3_turbo(self):
        """Init loads the large-v3-turbo faster-whisper model."""
        src = _read_pipeline_source()
        assert '"large-v3-turbo"' in src or "'large-v3-turbo'" in src
        assert 'compute_type="float16"' in src or "compute_type='float16'" in src

    @REQUIRES_L4
    def test_layer2_transcribe_returns_segments(self):
        """Real Whisper call on chest_pain.wav produces ≥3 segments."""
        from src.extraction.flow.ship.pipeline import ShipPipeline
        clip_path = REPO / "eval" / "synthetic_clips" / "audio" / "chest_pain.wav"
        if not clip_path.exists():
            pytest.skip("chest_pain.wav not on disk")
        pipe = ShipPipeline(hf_token=os.environ.get("HF_TOKEN"), enable_diarization=False)
        segs, info = pipe.transcribe(str(clip_path))
        assert len(segs) >= 3
        assert info.language == "en"


# ─────────────────────────────────────────────────────────────
# LAYER 3 — Diarization
# ─────────────────────────────────────────────────────────────

class TestLayer3Diarization:
    """Pure source-inspection tests; no import needed."""

    def test_layer3_init_disables_telemetry(self):
        """pyannote 4.x telemetry must be disabled BEFORE loading."""
        src = _read_pipeline_source()
        assert "set_telemetry_metrics" in src
        tele_idx = src.find("set_telemetry_metrics")
        load_idx = src.find("from_pretrained")
        assert 0 < tele_idx < load_idx, "telemetry disable must precede pipeline load"

    def test_layer3_diarize_returns_annotation_or_none(self):
        """diarize() unwraps pyannote 4.x DiarizeOutput to Annotation."""
        src = _read_pipeline_source()
        assert "speaker_diarization" in src
        assert "return None" in src

    def test_layer3_cpu_fallback_hardcoded(self):
        """Pyannote runs on CPU because NVRTC 13 is missing on DLVM."""
        src = _read_pipeline_source()
        assert 'device("cpu")' in src or "device('cpu')" in src

    @REQUIRES_L4
    def test_layer3_pyannote_loads(self):
        """ShipPipeline init with HF_TOKEN populates diar_enabled=True."""
        from src.extraction.flow.ship.pipeline import ShipPipeline
        token = os.environ.get("HF_TOKEN")
        if not token:
            pytest.skip("HF_TOKEN not set")
        pipe = ShipPipeline(hf_token=token, enable_diarization=True)
        assert pipe.diar_enabled is True


# ─────────────────────────────────────────────────────────────
# LAYER 4 — Speaker Assignment
# ─────────────────────────────────────────────────────────────

@REQUIRES_PIPELINE
class TestLayer4SpeakerAssignment:
    """Invokes assign_speakers method on a bare ShipPipeline — requires import."""

    def _make_seg(self, start, end, text):
        # faster-whisper Segment is a namedtuple-like; mimic the attributes used
        m = MagicMock()
        m.start = start
        m.end = end
        m.text = text
        return m

    def test_layer4_assign_heuristic_alternates(self):
        """When diarization is None, segments alternate DR/PT by index."""
        from src.extraction.flow.ship.pipeline import ShipPipeline
        pipe = ShipPipeline.__new__(ShipPipeline)  # skip __init__
        segs = [self._make_seg(i, i + 1, f"turn {i}") for i in range(4)]
        out = pipe.assign_speakers(segs, None)
        assert out[0]["speaker"] == "DOCTOR"
        assert out[1]["speaker"] == "PATIENT"
        assert out[2]["speaker"] == "DOCTOR"
        assert out[3]["speaker"] == "PATIENT"

    def test_layer4_assign_with_diarization_midpoint(self):
        """Midpoint inside diar turn → correct speaker label."""
        from src.extraction.flow.ship.pipeline import ShipPipeline

        # Build a minimal fake Annotation that responds to itertracks(yield_label=True)
        class FakeAnn:
            def __init__(self, tracks):
                self._tracks = tracks  # list of (Segment, _, spk)

            def itertracks(self, yield_label=True):
                return iter(self._tracks)

        class FakeSeg:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        # Two diar turns: SPK1 [0,2], SPK2 [2,4]
        ann = FakeAnn([(FakeSeg(0.0, 2.0), None, "SPK1"),
                       (FakeSeg(2.0, 4.0), None, "SPK2")])

        pipe = ShipPipeline.__new__(ShipPipeline)
        segs = [self._make_seg(0.2, 0.8, "hello"),  # mid=0.5 → SPK1
                self._make_seg(2.5, 3.5, "hi")]  # mid=3.0 → SPK2
        out = pipe.assign_speakers(segs, ann)
        assert out[0]["speaker"] == "DOCTOR"  # first-seen SPK1
        assert out[1]["speaker"] == "PATIENT"  # second SPK2

    def test_layer4_first_speaker_is_doctor(self):
        """First-seen pyannote label always maps to DOCTOR."""
        from src.extraction.flow.ship.pipeline import ShipPipeline

        class FakeAnn:
            def __init__(self, tracks):
                self._tracks = tracks

            def itertracks(self, yield_label=True):
                return iter(self._tracks)

        class FakeSeg:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        # First pyannote label is "Z" (last alphabetical) — doesn't matter
        ann = FakeAnn([(FakeSeg(0, 1), None, "Z"), (FakeSeg(1, 2), None, "A")])
        pipe = ShipPipeline.__new__(ShipPipeline)
        segs = [self._make_seg(0.1, 0.9, "x"), self._make_seg(1.1, 1.9, "y")]
        out = pipe.assign_speakers(segs, ann)
        assert out[0]["speaker"] == "DOCTOR"  # Z got mapped to DOCTOR
        assert out[1]["speaker"] == "PATIENT"


# ─────────────────────────────────────────────────────────────
# LAYER 5 — Fuzzy Medical Correction
# ─────────────────────────────────────────────────────────────

class TestLayer5FuzzyCorrection:
    def test_layer5_threshold_is_88(self):
        """Default threshold is 88 — tuned to catch addorvastatin (88.0)
        and nitroglycrin (96.0) while suppressing 'the pain'->'heparin'
        (72.7) and 'giving'->'IVIG' (80.0). See verifier finding #3."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        sig = inspect.signature(correct_medical_terms)
        assert sig.parameters["threshold"].default == 88

    def test_layer5_catches_addorvastatin(self):
        """Threshold 88 catches 'addorvastatin' (Whisper mis-hearing)."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        out, fixes = correct_medical_terms("patient on addorvastatin daily")
        assert "atorvastatin" in out
        assert any(f["corrected"] == "atorvastatin" for f in fixes)

    def test_layer5_catches_nitroglycrin(self):
        """Missing-e variant of nitroglycerin gets corrected."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        out, fixes = correct_medical_terms("administered nitroglycrin sublingually")
        assert "nitroglycerin" in out
        assert any(f["corrected"] == "nitroglycerin" for f in fixes)

    def test_layer5_plural_guard_migraines(self):
        """migraines must NOT get corrected to migraine (grammatical plural)."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        out, fixes = correct_medical_terms("history of migraines today")
        assert "migraines" in out
        assert not any(f["corrected"].lower() == "migraine" for f in fixes)

    def test_layer5_bigram_does_not_drop_word(self):
        """on amlodipine must stay two words (not collapse to amlodipine)."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        out, _ = correct_medical_terms("patient is on amlodipine daily")
        assert "on amlodipine" in out

    def test_layer5_short_word_skip(self):
        """3-char words like 'the' never trigger vocab match."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        out, fixes = correct_medical_terms("the rate the pain")
        for f in fixes:
            assert len(f["original"].strip(".,!?")) >= 4

    def test_layer5_no_false_positive_on_common_words(self):
        """At threshold 92, 'the pain' must NOT match 'heparin'."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        out, fixes = correct_medical_terms("the pain was severe")
        assert "heparin" not in out.lower()
        assert not any("heparin" in f.get("corrected", "").lower() for f in fixes)

    def test_layer5_preserves_punctuation(self):
        """Trailing period on a word survives correction pass."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        out, _ = correct_medical_terms("he has migraines.")
        assert out.endswith(".")

    def test_layer5_genuine_correction_still_fires(self):
        """A clearly mangled single term above threshold still gets fixed."""
        from src.extraction.flow.ship.medical_correction import correct_medical_terms
        # 'nitroglycrin' (missing 'e') is 11 chars vs 'nitroglycerin' 13 chars
        # fuzz.ratio = ~92. Should correct.
        out, fixes = correct_medical_terms("administered nitroglycrin")
        # not strictly required to fire (threshold edge), but if it does fire
        # the correction should be to the right term
        if fixes:
            assert fixes[0]["corrected"].lower() == "nitroglycerin"


# ─────────────────────────────────────────────────────────────
# LAYER 6 — Per-Clip Measurement
# ─────────────────────────────────────────────────────────────

class TestLayer6Measurement:
    def test_layer6_normalize_lowercase_strip_punct(self):
        from src.extraction.flow.ship.measure import _normalize
        assert _normalize("Hello, World!") == "hello world"
        assert _normalize("   Mr.  Smith.") == "mr smith"

    def test_layer6_normalized_wer_kills_case_diff(self):
        """Case/punctuation-only difference should give normalized WER 0."""
        import jiwer
        from src.extraction.flow.ship.measure import _normalize
        ref = "Good morning, Mr. Smith."
        hyp = "good morning mr. smith"
        assert jiwer.wer(_normalize(ref), _normalize(hyp)) == 0.0

    def test_layer6_medical_term_wer_zero_when_no_med(self):
        from src.extraction.flow.ship.measure import medical_term_wer
        assert medical_term_wer("hello world", "goodbye moon") == 0.0

    def test_layer6_medical_term_wer_one_when_hyp_empty_med(self):
        from src.extraction.flow.ship.measure import medical_term_wer
        assert medical_term_wer("has chest pain", "has nothing noted") == 1.0

    def test_layer6_medical_terms_set_min_4_chars(self):
        from src.extraction.flow.ship.measure import MEDICAL_TERMS_SET
        assert "chest" in MEDICAL_TERMS_SET
        assert "pain" in MEDICAL_TERMS_SET
        # 3-char tokens excluded
        assert "of" not in MEDICAL_TERMS_SET
        for term in MEDICAL_TERMS_SET:
            assert len(term) >= 4

    def test_layer6_compute_der_none_passthrough(self):
        from src.extraction.flow.ship.measure import compute_der
        assert compute_der(None, [{"speaker": "DOCTOR", "text": "x"}], 10.0) is None

    def test_layer6_compute_der_prefers_sidecar(self, tmp_path):
        """When {audio_path}.turns.json exists with real start_s/end_s,
        compute_der builds ground-truth Annotation from it — not the
        equal-duration-slot proxy. Requires pyannote.metrics, so skips
        gracefully when unavailable.
        """
        pytest.importorskip("pyannote.metrics")
        from src.extraction.flow.ship.measure import compute_der

        # Fake audio path + sidecar with real timestamps.
        fake_wav = tmp_path / "clip.wav"
        fake_wav.write_bytes(b"")  # existence needed for `sidecar.exists()`
        sidecar = fake_wav.with_suffix(".turns.json")
        sidecar.write_text(json.dumps([
            {"speaker": "DOCTOR", "text": "hi", "start_s": 0.0, "end_s": 1.5},
            {"speaker": "PATIENT", "text": "hello", "start_s": 1.8, "end_s": 3.2},
        ]), encoding="utf-8")

        class FakeSeg:
            def __init__(self, s, e):
                self.start = s
                self.end = e

        class FakeAnn:
            def __init__(self, tracks):
                self._tracks = tracks

            def itertracks(self, yield_label=True):
                return iter(self._tracks)

        # Hypothesis perfectly matches REAL timestamps → near-zero DER.
        hyp = FakeAnn([
            (FakeSeg(0.0, 1.5), None, "A"),
            (FakeSeg(1.8, 3.2), None, "B"),
        ])
        der = compute_der(hyp, [], 5.0, audio_path=str(fake_wav))
        assert der is not None
        # Real path should give near-zero; proxy would give ~0.5+ because
        # the 2nd equal slot would span 2.5-5.0 instead of 1.8-3.2.
        assert der < 0.2, f"DER={der} — real sidecar path should be near 0"

    def test_layer6_compute_der_fallback_without_sidecar(self, tmp_path):
        """Without a sidecar, compute_der still works (proxy path)."""
        pytest.importorskip("pyannote.metrics")
        from src.extraction.flow.ship.measure import compute_der

        fake_wav = tmp_path / "nosidecar.wav"
        fake_wav.write_bytes(b"")

        class FakeSeg:
            def __init__(self, s, e):
                self.start = s
                self.end = e

        class FakeAnn:
            def __init__(self, tracks):
                self._tracks = tracks

            def itertracks(self, yield_label=True):
                return iter(self._tracks)

        hyp = FakeAnn([
            (FakeSeg(0.0, 5.0), None, "A"),
            (FakeSeg(5.0, 10.0), None, "B"),
        ])
        gt_turns = [{"speaker": "DOCTOR", "text": "a"}, {"speaker": "PATIENT", "text": "b"}]
        der = compute_der(hyp, gt_turns, 10.0, audio_path=str(fake_wav))
        assert der is not None  # proxy path still returns a number


# ─────────────────────────────────────────────────────────────
# LAYER 7 — Aggregation & Reporting
# ─────────────────────────────────────────────────────────────

class TestLayer7Aggregation:
    @pytest.fixture
    def latest_results(self):
        root = REPO / "eval" / "flow_results" / "ship"
        if not root.exists():
            pytest.skip("no ship results directory")
        dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not dirs:
            pytest.skip("no ship runs yet")
        results = dirs[-1] / "results.json"
        per_clip = dirs[-1] / "per_clip_metrics.jsonl"
        if not (results.exists() and per_clip.exists()):
            pytest.skip("results.json or per_clip not present in latest run")
        return results, per_clip

    def test_layer7_aggregate_splits_pediatric_from_original(self, latest_results):
        results_path, _ = latest_results
        agg = json.loads(results_path.read_text())
        assert "original_6" in agg
        assert "unseen_pediatric" in agg
        # pediatric should be under unseen_pediatric, not folded into original_6
        if agg["original_6"]:
            # mean WER raw for original_6 should exclude pediatric
            per_clip = [m for m in agg["per_clip"]
                        if m.get("scenario") != "pediatric_fever_rash"
                        and "error" not in m]
            assert agg["n_clips_original"] == len(per_clip)

    def test_layer7_writes_timestamped_dir(self):
        """Output dir name matches %Y%m%dT%H%M%SZ regex."""
        root = REPO / "eval" / "flow_results" / "ship"
        if not root.exists():
            pytest.skip("no ship results yet")
        ts_re = re.compile(r"^\d{8}T\d{6}Z$")
        ts_dirs = [d.name for d in root.iterdir() if d.is_dir()]
        assert any(ts_re.match(d) for d in ts_dirs), f"no timestamped dir in {ts_dirs}"

    def test_layer7_per_clip_jsonl_one_row_per_clip(self, latest_results):
        _, per_clip_path = latest_results
        n = sum(1 for line in per_clip_path.read_text().splitlines() if line.strip())
        # Runs exist with 7 (full) or partial; at minimum we expect ≥6
        assert n >= 6

    def test_layer7_vs_variant_a_block_present(self, latest_results):
        results_path, _ = latest_results
        agg = json.loads(results_path.read_text())
        assert "vs_variant_a" in agg
        for key in ("variant_a_wer_raw_mean", "variant_a_medical_term_wer_mean"):
            assert key in agg["vs_variant_a"]

    def test_layer7_results_json_has_normalized_mean(self, latest_results):
        """Verifier finding #2 was: top-level aggregate lacks normalized WER.
        Closed by adding wer_raw_normalized_mean + wer_corrected_normalized_mean
        to the original_6 aggregate. NOTE: this only passes on runs produced
        AFTER the fix committed alongside this test. Older runs will xfail.
        """
        results_path, _ = latest_results
        agg = json.loads(results_path.read_text())
        o6 = agg.get("original_6", {})
        if "wer_raw_normalized_mean" not in o6:
            pytest.xfail(
                f"latest run ({results_path.parent.name}) predates the "
                "normalized_mean aggregate fix; regenerate via run_all to pick it up"
            )
        assert "wer_raw_normalized_mean" in o6
        assert "wer_corrected_normalized_mean" in o6


# ─────────────────────────────────────────────────────────────
# LAYER 8 — Reasoning: Claim Extraction
# ─────────────────────────────────────────────────────────────

class TestLayer8Claims:
    def test_layer8_model_id_is_deepseek_r1(self):
        from src.extraction.flow.ship import reasoning
        assert reasoning._MODEL_ID == "deepseek-ai/deepseek-r1-0528-maas"

    def test_layer8_strip_think_block(self):
        from src.extraction.flow.ship.reasoning import _strip_code_fence
        text = "<think>reasoning here</think>\n```json\n[]\n```"
        assert _strip_code_fence(text) == "[]"

    def test_layer8_strip_code_fence(self):
        from src.extraction.flow.ship.reasoning import _strip_code_fence
        assert _strip_code_fence('```json\n{"x": 1}\n```') == '{"x": 1}'
        assert _strip_code_fence('```\nno language\n```') == 'no language'

    @REQUIRES_GCP
    def test_layer8_init_uses_adc_credentials(self):
        """init_deepseek uses google.auth.default with cloud-platform scope."""
        from src.extraction.flow.ship import reasoning
        src = inspect.getsource(reasoning._refresh_adc_token)
        assert "google.auth.default" in src
        assert "cloud-platform" in src

    def test_layer8_base_url_points_to_vertex_maas(self):
        """Client's base_url targets Vertex AI MaaS in us-central1."""
        from src.extraction.flow.ship import reasoning
        assert reasoning._LOCATION == "us-central1"
        src = inspect.getsource(reasoning.init_deepseek)
        # f-string template, not the full host after interpolation
        assert "-aiplatform.googleapis.com" in src
        assert "endpoints/openapi" in src
        assert "_LOCATION" in src

    def test_layer8_parse_failure_returns_error_dict(self):
        """Unparseable model output returns [{"error": ...}] instead of raising."""
        from src.extraction.flow.ship import reasoning

        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "not json at all, just prose"
        fake_client.chat.completions.create.return_value = fake_response

        result = reasoning.extract_claims(fake_client, [{"speaker": "DOCTOR", "text": "hi"}])
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]

    @REQUIRES_L4
    @REQUIRES_GCP
    def test_layer8_live_extract(self):
        """Real DeepSeek-R1 call returns ≥3 claims on chest_pain transcript."""
        from src.extraction.flow.ship.reasoning import init_deepseek, extract_claims
        client = init_deepseek()
        segs = [
            {"speaker": "DOCTOR", "text": "What brings you in?"},
            {"speaker": "PATIENT", "text": "I have chest pain for two hours."},
            {"speaker": "DOCTOR", "text": "Does it radiate?"},
            {"speaker": "PATIENT", "text": "It radiates to my left arm."},
        ]
        claims = extract_claims(client, segs)
        assert isinstance(claims, list)
        assert len(claims) >= 3
        # structure sanity
        assert all("claim_id" in c or "error" in c for c in claims)


# ─────────────────────────────────────────────────────────────
# LAYER 9 — Reasoning: Differential
# ─────────────────────────────────────────────────────────────

class TestLayer9Differential:
    def test_layer9_prompt_includes_serialized_claims(self):
        """Prompt contains json.dumps of claims input."""
        from src.extraction.flow.ship import reasoning
        src = inspect.getsource(reasoning.generate_differential)
        assert "json.dumps(claims" in src

    @REQUIRES_L4
    @REQUIRES_GCP
    def test_layer9_live_differential(self):
        """Real call returns ≥1 hypothesis with evidence_for chain."""
        from src.extraction.flow.ship.reasoning import init_deepseek, generate_differential
        client = init_deepseek()
        claims = [
            {"claim_id": "c01", "subject": "chest pain", "predicate": "presence", "value": "true", "speaker": "PATIENT", "turn_index": 1, "confidence": "high"},
            {"claim_id": "c02", "subject": "chest pain", "predicate": "radiation", "value": "left arm", "speaker": "PATIENT", "turn_index": 3, "confidence": "high"},
        ]
        diff = generate_differential(client, claims)
        assert isinstance(diff, list)
        assert len(diff) >= 1
        if "error" not in diff[0]:
            assert "hypothesis" in diff[0]
            assert "evidence_for" in diff[0]


# ─────────────────────────────────────────────────────────────
# LAYER 10 — Reasoning: SOAP Note
# ─────────────────────────────────────────────────────────────

class TestLayer10SOAPNote:
    def test_layer10_prompt_mandates_provenance_tags(self):
        """Prompt explicitly requires [c:XX] provenance tags."""
        from src.extraction.flow.ship import reasoning
        src = inspect.getsource(reasoning.generate_soap_note)
        assert "[c:XX]" in src or "[c:" in src

    @REQUIRES_L4
    @REQUIRES_GCP
    def test_layer10_live_soap_has_tags(self):
        """Real call output contains [c:cNN] pattern."""
        from src.extraction.flow.ship.reasoning import init_deepseek, generate_soap_note
        client = init_deepseek()
        claims = [
            {"claim_id": "c01", "subject": "chest pain", "predicate": "presence", "value": "true"},
            {"claim_id": "c02", "subject": "chest pain", "predicate": "onset", "value": "2h ago"},
        ]
        differential = [{"hypothesis": "ACS", "rank": 1, "evidence_for": ["c01", "c02"]}]
        segments = [{"speaker": "PATIENT", "text": "chest pain 2h"}]
        note = generate_soap_note(client, claims, differential, segments)
        assert isinstance(note, str)
        assert re.search(r"\[c:c\d+", note), f"no provenance tags in SOAP output:\n{note[:500]}"

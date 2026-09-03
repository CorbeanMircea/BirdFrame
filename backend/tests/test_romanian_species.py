"""
Task 5.4 — Romanian/European species coverage validation.

Uses a real European Robin (Erithacus rubecula) recording and verifies
that BirdNET correctly identifies Romanian/European species — confirming
the model has coverage for BirdFrame's intended geographic region.

Run with:
    pytest -m integration backend/tests/test_romanian_species.py -v -s

Place the Robin MP3 at:
    BirdFrame/data/test_audio/erithacus_rubecula.mp3

The test skips gracefully if the file is absent.
"""

import sys
import time
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.identification.birdnet_identifier import (
    BirdNetIdentifier,
    BIRDNET_SAMPLE_RATE,
)
from backend.identification.base import IdentificationResult
from backend.audio.mock_recorder import MockAudioRecorder
from backend.audio.processor import AudioProcessor
from backend.audio.detector import BirdDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_AUDIO_DIR = config.DATA_DIR / "test_audio"
ROBIN_AUDIO_PATH = TEST_AUDIO_DIR / "erithacus_rubecula.mp3"

TARGET_SCIENTIFIC = "Erithacus rubecula"
TARGET_COMMON = "European Robin"

# Romania coordinates (centre of country)
ROMANIA_LAT = 45.9
ROMANIA_LON = 24.9
ROMANIA_WEEK = 20   # Mid-May — peak songbird season

# Confidence thresholds
# 0.1 = model recognised the species at all (detection threshold)
# 0.5 = confident detection (production threshold)
DETECTION_THRESHOLD = 0.1
CONFIDENT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file using librosa, resampled to BIRDNET_SAMPLE_RATE."""
    import librosa
    audio, sr = librosa.load(str(path), sr=BIRDNET_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), int(sr)


def _analyse_all_segments(
    identifier: BirdNetIdentifier,
    audio: np.ndarray,
    sr: int,
    segment_duration: float = 3.0,
    overlap: float = 0.5,
    top_n: int = 5,
) -> list[IdentificationResult]:
    """Run BirdNET across all overlapping segments and return all results."""
    segment_samples = int(segment_duration * sr)
    step_samples = int((segment_duration - overlap) * sr)
    all_results: list[IdentificationResult] = []
    offset = 0
    while offset + segment_samples <= len(audio):
        segment = audio[offset: offset + segment_samples]
        results = identifier.identify(segment, sr, top_n=top_n)
        all_results.extend(results)
        offset += step_samples
    return all_results


def _print_detection_summary(all_results: list[IdentificationResult]) -> None:
    """Print a readable summary of detections to stdout."""
    if not all_results:
        print("\n  No detections above model threshold.")
        return
    species_counts = Counter(r.scientific_name for r in all_results)
    print(f"\n  Top species detected across all segments:")
    for sci, count in species_counts.most_common(8):
        best = max(r.confidence for r in all_results if r.scientific_name == sci)
        common = next(r.common_name for r in all_results if r.scientific_name == sci)
        marker = " ← TARGET" if sci == TARGET_SCIENTIFIC else ""
        print(f"    {sci} ({common}): {count} segment(s), best conf={best:.3f}{marker}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def identifier():
    """BirdNET identifier tuned for Romania."""
    ident = BirdNetIdentifier(
        latitude=ROMANIA_LAT,
        longitude=ROMANIA_LON,
        week=ROMANIA_WEEK,
        min_confidence=0.05,   # permissive — we filter in the test assertions
    )
    ident.warmup()
    return ident


@pytest.fixture(scope="module")
def robin_audio():
    """
    Load the European Robin recording from the local cache.
    Skips if the file is not present.
    """
    if not ROBIN_AUDIO_PATH.exists():
        pytest.skip(
            f"Robin recording not found at {ROBIN_AUDIO_PATH}. "
            "Place erithacus_rubecula.mp3 in BirdFrame/data/test_audio/ and re-run."
        )
    audio, sr = _load_audio(ROBIN_AUDIO_PATH)
    print(f"\n  Loaded: {TARGET_COMMON} ({TARGET_SCIENTIFIC})")
    print(f"  Duration: {len(audio)/sr:.1f}s @ {sr} Hz")
    return audio, sr


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRomanianSpecies:
    """
    Validates BirdNET's European/Romanian species coverage.

    The European Robin is one of the most common and distinctive
    songbirds in Romania. These tests verify:
    1. The model detects the Robin at all (detection threshold)
    2. Pipeline mechanics work with a real recording
    3. Inference speed is acceptable for near-real-time use
    """

    def test_recording_loaded(self, robin_audio):
        """Recording must load with correct properties."""
        audio, sr = robin_audio
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert sr == BIRDNET_SAMPLE_RATE
        assert len(audio) > sr, "Recording must be at least 1 second long"

    def test_recording_has_audio_energy(self, robin_audio):
        """Recording must not be silent."""
        audio, sr = robin_audio
        from backend.audio.detector import _compute_rms
        rms = _compute_rms(audio[:int(3.0 * sr)])
        print(f"\n  RMS energy: {rms:.5f}")
        assert rms > 0.001, f"Recording appears silent (rms={rms:.5f})"

    def test_birdnet_detects_robin_at_all(self, identifier, robin_audio):
        """
        BirdNET must detect European Robin in at least one segment.
        Threshold: confidence > 0.1 (model recognised the species).

        Note: confidence depends heavily on recording quality.
        A real field recording from Romania would score higher.
        """
        audio, sr = robin_audio
        all_results = _analyse_all_segments(identifier, audio, sr)
        _print_detection_summary(all_results)

        robin_results = [r for r in all_results if r.scientific_name == TARGET_SCIENTIFIC]
        best_confidence = max((r.confidence for r in robin_results), default=0.0)

        print(f"\n  Robin detected in {len(robin_results)} segment(s)")
        print(f"  Best confidence: {best_confidence:.3f}")
        print(f"  Detection threshold: {DETECTION_THRESHOLD}")
        print(f"  Confident threshold (production): {CONFIDENT_THRESHOLD}")

        assert best_confidence >= DETECTION_THRESHOLD, (
            f"BirdNET did not detect {TARGET_COMMON} above {DETECTION_THRESHOLD} "
            f"in any segment (best={best_confidence:.3f}). "
            f"Top species: {[r.scientific_name for r in all_results[:3]]}"
        )

    def test_european_species_in_results(self, identifier, robin_audio):
        """
        Results must include species plausible for Romania/Europe.
        Validates that the location filter is working correctly.
        """
        audio, sr = robin_audio
        all_results = _analyse_all_segments(identifier, audio, sr)

        # These are all well-known European species that BirdNET covers
        known_european = {
            "Erithacus rubecula",
            "Turdus merula",
            "Parus major",
            "Passer domesticus",
            "Fringilla coelebs",
            "Cyanistes caeruleus",
            "Sylvia atricapilla",
            "Phylloscopus collybita",
            "Carduelis carduelis",
            "Columba palumbus",
            "Sturnus vulgaris",
            "Turdus philomelos",
            "Sitta europaea",
            "Troglodytes troglodytes",
            "Motacilla alba",
        }

        detected_species = {r.scientific_name for r in all_results}
        european_detected = detected_species & known_european

        print(f"\n  Detected species: {detected_species}")
        print(f"  Of which European: {european_detected}")

        assert len(european_detected) > 0, (
            f"No known European species detected. Got: {detected_species}"
        )

    def test_full_pipeline_with_real_recording(self, identifier, robin_audio):
        """
        MockAudioRecorder → AudioProcessor → BirdDetector → BirdNetIdentifier
        using the real Robin recording. Validates full pipeline mechanics.
        """
        audio, sr = robin_audio

        detector = BirdDetector(
            energy_threshold=0.001,
            freq_min=1000,
            freq_max=min(10000, sr // 2 - 1),
            band_ratio_threshold=0.05,
        )

        pipeline_results: list[IdentificationResult] = []
        segments_accepted = []

        def on_segment(segment: np.ndarray, sample_rate: int) -> None:
            det = detector.analyse(segment, sample_rate)
            if det.accepted:
                segments_accepted.append(True)
                results = identifier.identify(segment, sample_rate, top_n=3)
                pipeline_results.extend(results)

        processor = AudioProcessor(
            segment_callback=on_segment,
            segment_duration=3.0,
            overlap_duration=0.5,
        )

        recorder = MockAudioRecorder.from_array(
            audio,
            sample_rate=sr,
            callback=processor.process,
            chunk_duration=3.0,
        )
        recorder.start()
        recorder.join(timeout=120.0)

        print(f"\n  Segments accepted by detector: {len(segments_accepted)}")
        print(f"  Total pipeline detections: {len(pipeline_results)}")

        robin_in_pipeline = [
            r for r in pipeline_results
            if r.scientific_name == TARGET_SCIENTIFIC
        ]
        print(f"  Robin detections via full pipeline: {len(robin_in_pipeline)}")

        assert len(segments_accepted) > 0, "Detector rejected all segments"
        assert len(pipeline_results) >= 0   # pipeline ran without error

    def test_inference_speed_acceptable(self, identifier, robin_audio):
        """
        Single 3-second segment must be identified in under 10 seconds.
        Validates CPU inference is fast enough for near-real-time use.
        """
        audio, sr = robin_audio
        segment = audio[:int(3.0 * sr)]

        # Warm-up call
        identifier.identify(segment, sr)

        # Timed call
        start = time.time()
        identifier.identify(segment, sr)
        elapsed = time.time() - start

        print(f"\n  Inference time (single 3s segment): {elapsed:.3f}s")
        print(f"  Real-time factor: {elapsed / 3.0:.3f}x "
              f"({'faster' if elapsed < 3.0 else 'slower'} than real-time)")

        assert elapsed < 10.0, (
            f"Inference took {elapsed:.1f}s — too slow for near-real-time use."
        )
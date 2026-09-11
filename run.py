"""
BirdFrame — live pipeline runner.

Starts the full audio detection pipeline:
    Microphone → AudioProcessor → BirdDetector → BirdNetIdentifier
    → DetectionService → SQLite → (API serves results to dashboard)

Usage:
    python run.py                        # use default microphone, static artwork
    python run.py --generated-artwork    # use ComfyUI + Flux.1 for artwork
    python run.py --device 5             # use specific device index
    python run.py --mock                 # use MockBirdIdentifier (no model)
    python run.py --list-devices         # print available microphones

Run the API server separately:
    python -m backend.api.main

Run ComfyUI (for generated artwork):
    cd C:\\Users\\Gaming_PC\\ComfyUI_windows_portable
    .\\run_nvidia_gpu.bat

Then open the dashboard:
    http://localhost:3000
"""

import sys
import time
import signal
import logging
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("birdframe")

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("pydub").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="BirdFrame live audio pipeline"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Microphone device index (default: system default)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockBirdIdentifier instead of BirdNET (for testing)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--min-confidence", type=float,
        default=config.IDENTIFIER_MIN_CONFIDENCE,
        help=f"Minimum confidence (default: {config.IDENTIFIER_MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Stop automatically after this many seconds",
    )
    parser.add_argument(
        "--location", action="store_true",
        help="Enable Romanian location filter (lat=45.9, lon=24.9)",
    )
    parser.add_argument(
        "--generated-artwork", action="store_true",
        help=(
            "Use GeneratedArtworkProvider (ComfyUI + Flux.1). "
            "ComfyUI must be running at http://127.0.0.1:8188"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device listing
# ---------------------------------------------------------------------------

def list_devices():
    from backend.audio.recorder import print_input_devices
    print_input_devices()


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def build_identifier(args):
    if args.mock:
        logger.info("Using MockBirdIdentifier (sequential mode)")
        from backend.identification.mock_identifier import MockBirdIdentifier
        return MockBirdIdentifier(mode="sequential", fixed_confidence=0.85)

    logger.info("Loading BirdNET model…")
    from backend.identification.birdnet_identifier import BirdNetIdentifier

    lat, lon, week = None, None, None
    if args.location:
        import datetime
        lat, lon = 45.9, 24.9
        week = int(datetime.date.today().strftime("%W")) + 1
        logger.info(
            "Location filter: Romania (lat=%.1f lon=%.1f week=%d)",
            lat, lon, week,
        )

    return BirdNetIdentifier(
        latitude=lat,
        longitude=lon,
        week=week,
        min_confidence=0.05,
    )


def build_service(identifier, args):
    from backend.detection.service import DetectionService
    from backend.audio.detector import BirdDetector

    detector = BirdDetector(
        energy_threshold=config.DETECTOR_ENERGY_THRESHOLD,
        freq_min=config.DETECTOR_FREQ_MIN,
        freq_max=config.DETECTOR_FREQ_MAX,
        band_ratio_threshold=0.05,
    )

    return DetectionService(
        identifier=identifier,
        detector=detector,
        min_confidence=args.min_confidence,
        segment_duration=config.AUDIO_CHUNK_DURATION,
        overlap_duration=config.AUDIO_CHUNK_OVERLAP,
        target_sample_rate=48_000,
        enable_grouping=True,
    )


def build_recorder(service, args):
    from backend.audio.recorder import AudioRecorder

    device = args.device if args.device is not None else config.AUDIO_DEVICE_INDEX

    return AudioRecorder(
        callback=service.handle_chunk,
        sample_rate=config.AUDIO_SAMPLE_RATE,
        chunk_duration=config.AUDIO_CHUNK_DURATION,
        channels=config.AUDIO_CHANNELS,
        device=device,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    # Set artwork backend env var before importing anything that reads it
    if args.generated_artwork:
        os.environ["BIRDFRAME_ARTWORK"] = "generated"
        # Reload config to pick up the env var
        import importlib
        importlib.reload(config)

    artwork_backend = "generated" if args.generated_artwork else "static"

    print("\n" + "=" * 60)
    print("  BirdFrame — Live Detection Pipeline")
    print("=" * 60)
    print(f"  Identifier : {'Mock (sequential)' if args.mock else 'BirdNET'}")
    print(f"  Device     : {args.device if args.device is not None else 'system default'}")
    print(f"  Sample rate: {config.AUDIO_SAMPLE_RATE} Hz")
    print(f"  Chunk size : {config.AUDIO_CHUNK_DURATION}s")
    print(f"  Overlap    : {config.AUDIO_CHUNK_OVERLAP}s")
    print(f"  Min conf.  : {args.min_confidence}")
    print(f"  Location   : {'Romania' if args.location else 'disabled'}")
    print(f"  Artwork    : {artwork_backend}")
    print(f"  Duration   : {args.duration or 'unlimited'}")
    print("=" * 60)
    print("  Dashboard  : http://localhost:3000")
    print("  API docs   : http://127.0.0.1:8000/docs")
    if args.generated_artwork:
        print("  ComfyUI    : http://127.0.0.1:8188")
    print("=" * 60)
    print("  Press Ctrl+C to stop.\n")

    if args.generated_artwork:
        # Verify ComfyUI is reachable before starting
        from backend.artwork.generated_provider import GeneratedArtworkProvider
        probe = GeneratedArtworkProvider()
        if not probe._ping_comfyui():
            print("  ✗ ERROR: ComfyUI is not running at http://127.0.0.1:8188")
            print("  Start ComfyUI first:")
            print(r"    cd C:\Users\Gaming_PC\ComfyUI_windows_portable")
            print(r"    .\run_nvidia_gpu.bat")
            return
        print("  ✓ ComfyUI reachable\n")

    identifier = build_identifier(args)
    service = build_service(identifier, args)
    recorder = build_recorder(service, args)

    shutdown_requested = False

    def _shutdown(sig, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            print("\n\nShutting down…")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Starting detection service…")
    service.start()

    logger.info("Starting audio recorder…")
    try:
        recorder.start()
    except Exception as exc:
        logger.error("Failed to start recorder: %s", exc)
        logger.info("Try --list-devices to see available microphones.")
        service.stop()
        return

    logger.info("Listening for birds… 🎙️")

    start_time = time.time()
    last_stats_time = start_time

    try:
        while not shutdown_requested:
            time.sleep(1.0)

            now = time.time()
            if now - last_stats_time >= 30:
                stats = service.get_stats()
                elapsed = int(now - start_time)
                print(
                    f"\n  [{elapsed:>4}s] "
                    f"chunks={stats['chunks_received']}  "
                    f"analysed={stats['segments_analysed']}  "
                    f"accepted={stats['segments_accepted']}  "
                    f"saved={stats['detections_saved']}  "
                    f"events={stats['events_created_or_extended']}"
                )
                last_stats_time = now

            if args.duration and (time.time() - start_time) >= args.duration:
                logger.info("Duration reached — stopping.")
                break

    finally:
        recorder.stop()
        service.stop()

        stats = service.get_stats()
        elapsed = int(time.time() - start_time)
        print("\n" + "=" * 60)
        print("  Session Summary")
        print("=" * 60)
        print(f"  Runtime          : {elapsed}s")
        print(f"  Chunks received  : {stats['chunks_received']}")
        print(f"  Segments analysed: {stats['segments_analysed']}")
        print(f"  Segments accepted: {stats['segments_accepted']}")
        print(f"  Detections saved : {stats['detections_saved']}")
        print(f"  Events created   : {stats['events_created_or_extended']}")
        print(f"  Artwork backend  : {artwork_backend}")
        print("=" * 60)
        print(f"  Dashboard        : http://localhost:3000")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
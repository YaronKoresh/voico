import argparse
import logging
import os
import sys
from pathlib import Path

from .converter import VoiceConverter
from .core.config import ConversionQuality
from .core.errors import ProfileQualityError, ValidationError, VoicoError
from .quality.diagnostic import DiagnosticLogger
from .utils.audio_io import get_audio_info

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voico: Modular Voice Converter"
    )

    parser.add_argument(
        "input_file",
        help="Path to input audio file (Source Voice)",
    )
    parser.add_argument(
        "-t",
        "--target",
        help="Path to target audio file (Target Voice to mimic)",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output file",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--pitch",
        type=float,
        default=0.0,
        help="Manual pitch shift in semitones (overridden if Target is provided)",
    )
    parser.add_argument(
        "-f",
        "--formant",
        type=float,
        default=1.0,
        help="Manual formant shift factor (overridden if Target is provided)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=str,
        choices=[e.value for e in ConversionQuality],
        default="balanced",
        help="Processing quality preset",
    )
    parser.add_argument(
        "-b",
        "--bit-depth",
        type=int,
        choices=[16, 32],
        default=16,
        help="Output bit depth (16=int16 PCM, 32=float32)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display audio file info and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def print_audio_info(path: str) -> None:
    info = get_audio_info(path)
    print(f"File:        {info['path']}")
    print(f"Format:      {info.get('format', 'unknown')}")
    size = int(str(info.get("file_size_bytes", 0)))
    if size >= 1048576:
        print(f"Size:        {size / 1048576:.1f} MB")
    else:
        print(f"Size:        {size / 1024:.1f} KB")
    print(f"Sample Rate: {info.get('sample_rate', 'unknown')} Hz")
    print(f"Channels:    {info.get('channels', 'unknown')}")
    print(f"Frames:      {info.get('frames', 'unknown')}")
    duration = info.get("duration_seconds")
    if duration is not None:
        print(f"Duration:    {float(str(duration)):.3f}s")
    subtype = info.get("subtype")
    if subtype is not None:
        print(f"Subtype:     {subtype}")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not os.path.exists(args.input_file):
        logger.error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)

    if args.info:
        try:
            print_audio_info(args.input_file)
        except Exception as e:
            logger.error(f"Failed to read info: {e}")
            sys.exit(1)
        return

    if args.target and not os.path.exists(args.target):
        logger.error(f"Target file '{args.target}' not found.")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input_file)
        if args.target:
            target_name = Path(args.target).stem
            output_path = f"{base}_to_{target_name}{ext}"
        else:
            output_path = f"{base}_shifted_p{args.pitch}_f{args.formant}{ext}"

    try:
        quality = ConversionQuality(args.quality)
        converter = VoiceConverter(quality)

        diagnostic = DiagnosticLogger(args.input_file)

        converter.process(
            input_path=args.input_file,
            output_path=output_path,
            pitch_shift=args.pitch,
            formant_shift=args.formant,
            target_path=args.target,
            bit_depth=args.bit_depth,
            diagnostic_logger=diagnostic,
        )

        if args.verbose:
            diagnostic.print_summary()

    except ProfileQualityError as e:
        logger.error(f"Profile quality check failed: {e.message}")
        if e.recovery_suggestions:
            logger.error("Suggestions to improve:")
            for suggestion in e.recovery_suggestions:
                logger.error(f"  • {suggestion}")
        sys.exit(1)
    except VoicoError as e:
        logger.error(f"Conversion error: {e.message if hasattr(e, 'message') else str(e)}")
        if hasattr(e, 'recovery_suggestions') and e.recovery_suggestions:
            logger.error("Suggestions:")
            for suggestion in e.recovery_suggestions:
                logger.error(f"  • {suggestion}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
import logging
import os
import sys
from pathlib import Path

from .converter import VoiceConverter
from .core.config import ConversionQuality

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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not os.path.exists(args.input_file):
        logger.error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)

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
        converter.process(
            input_path=args.input_file,
            output_path=output_path,
            pitch_shift=args.pitch,
            formant_shift=args.formant,
            target_path=args.target,
        )
    except Exception as e:
        logging.error(f"Conversion failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

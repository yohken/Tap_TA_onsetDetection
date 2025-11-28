"""
TextGrid file viewer for command-line display.

This script displays the contents of a TextGrid file, including:
1. TextGrid structure (tier names, types, interval/point counts)
2. Phoneme intervals from the 'phones' tier (start - end: label format)

Usage:
    python display_textgrid.py your_speech.TextGrid
    python display_textgrid.py --help

Target Python version: 3.10+
Dependencies: textgrid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from textgrid import TextGrid, IntervalTier, PointTier


def display_textgrid_structure(tg: TextGrid) -> None:
    """
    Display the structure of a TextGrid file.

    Args:
        tg: TextGrid object to display.

    Prints tier names, types, and interval/point counts.
    """
    print("TextGrid structure:")
    for tier in tg.tiers:
        tier_type = type(tier).__name__
        if isinstance(tier, IntervalTier):
            count = len(tier)
            count_label = f"{count} intervals"
        elif isinstance(tier, PointTier):
            count = len(tier)
            count_label = f"{count} points"
        else:
            count_label = "unknown"
        print(f"  - Tier: {tier.name} ({tier_type}, {count_label})")


def display_phones_tier(tg: TextGrid, tier_name: str = "phones") -> None:
    """
    Display the intervals from the specified tier.

    Args:
        tg: TextGrid object to search.
        tier_name: Name of the tier to display (default: "phones").

    Prints each interval's start time, end time, and label.
    """
    # Find the specified tier
    tier = None
    for t in tg.tiers:
        if t.name == tier_name:
            tier = t
            break

    if tier is None:
        print(f"\nTier '{tier_name}' not found in TextGrid.")
        return

    print(f"\nPhoneme intervals ({tier_name} tier):")

    if isinstance(tier, IntervalTier):
        for interval in tier:
            label = interval.mark if interval.mark else ""
            print(f"  {interval.minTime:.2f} - {interval.maxTime:.2f}: {label}")
    elif isinstance(tier, PointTier):
        for point in tier:
            label = point.mark if point.mark else ""
            print(f"  {point.time:.2f}: {label}")
    else:
        print(f"  Unknown tier type: {type(tier).__name__}")


def main() -> int:
    """
    Main entry point for the TextGrid viewer.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Display TextGrid file contents (tier structure and phoneme intervals).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python display_textgrid.py your_speech.TextGrid
  python display_textgrid.py speech.TextGrid --tier words
        """
    )

    parser.add_argument(
        "tg_path",
        type=str,
        help="Path to the TextGrid file to display"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="phones",
        help="Name of the tier to display intervals for (default: phones)"
    )

    args = parser.parse_args()

    # Validate input file exists
    tg_path = Path(args.tg_path)
    if not tg_path.exists():
        print(f"Error: TextGrid file not found: {tg_path}")
        return 1

    # Load and display TextGrid
    try:
        tg = TextGrid.fromFile(str(tg_path))
    except Exception as e:
        print(f"Error loading TextGrid file: {e}")
        return 1

    print(f"---")
    display_textgrid_structure(tg)
    display_phones_tier(tg, tier_name=args.tier)

    return 0


if __name__ == "__main__":
    sys.exit(main())

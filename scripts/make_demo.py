"""Generate synthetic hourly inputs for a trial run."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_frame(rows):
    """Generate deterministic synthetic load observations.

    Args:
        rows (int): Number of consecutive hourly rows.

    Returns:
        DataFrame: Synthetic target, weather and calendar.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(
        "2024-01-01", periods=rows, freq="h"
    )
    hours = dates.hour.to_numpy()
    temperature = (
        15 + 8 * np.sin(2 * np.pi * (hours - 6) / 24)
        + rng.normal(0, 0.5, rows)
    )
    load = (
        50 + 9 * np.sin(2 * np.pi * hours / 24)
        + 0.4 * temperature + rng.normal(0, 0.8, rows)
    )
    return pd.DataFrame({
        "timestamp": dates,
        "load": load,
        "temperature": temperature,
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "is_weekend": (dates.dayofweek >= 5).astype(int),
    })


def main():
    """Parse options and write a local synthetic CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("examples/demo.csv"),
    )
    parser.add_argument("--rows", type=int, default=336)
    args = parser.parse_args()
    if args.rows < 64:
        parser.error("--rows must be at least 64")
    frame = make_frame(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output, index=False, float_format="%.6f"
    )
    print(
        f"Saved {len(frame)} synthetic rows "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()

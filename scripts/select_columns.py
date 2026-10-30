"""Select only configured inputs from a prepared table."""

import argparse
import json
from pathlib import Path

import pandas as pd


def selected_columns(config):
    """List required columns without duplicate targets.

    Args:
        config (dict): Dataset configuration section.

    Returns:
        list[str]: Ordered time, input and target names.
    """
    time_columns = (
        [config["time_column"]]
        if config.get("time_column") else []
    )
    return list(dict.fromkeys(
        time_columns + config["ts_features"]
        + config["static_features"] + [config["target"]]
    ))


def main():
    """Read a prepared table and export required columns."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("--config", "--input", "--output"):
        parser.add_argument(
            option, required=True, type=Path
        )
    args = parser.parse_args()
    config = json.loads(
        args.config.read_text(encoding="utf-8-sig")
    )["data"]
    frame = (
        pd.read_excel(args.input)
        if args.input.suffix.lower() == ".xlsx"
        else pd.read_csv(args.input)
    )
    columns = selected_columns(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(args.output, index=False)
    print(
        f"Saved {len(frame)} rows and {len(columns)} "
        f"columns to {args.output}"
    )


if __name__ == "__main__":
    main()

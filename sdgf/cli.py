"""Provide command-line entry points for SDGF workflows."""

import argparse
import os
from pathlib import Path


def positive_int(value):
    """Parse a strictly positive command-line integer.

    Args:
        value (str): Raw argument text.

    Returns:
        int: Parsed positive value.

    Raises:
        ArgumentTypeError: The parsed value is not positive.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "Must be a positive integer."
        )
    return parsed


def _prepare_parser(commands):
    """Register source preparation options.

    Args:
        commands (SubParsersAction): Command collection.
    """
    command = commands.add_parser(
        "prepare",
        help="Prepare a benchmark dataset from its source "
        "table",
    )
    command.add_argument(
        "--dataset", required=True,
        choices=("load_1", "load_2", "load_3"),
    )
    command.add_argument(
        "--input", required=True, type=Path
    )
    command.add_argument(
        "--output", required=True, type=Path
    )
    command.add_argument(
        "--date-format",
        help="Override timestamps with an explicit "
        "strptime format",
    )
    command.add_argument(
        "--encoding", default="utf-8-sig",
        help="Input CSV encoding",
    )


def _training_parser(commands):
    """Register training configuration and override options.

    Args:
        commands (SubParsersAction): Command collection.
    """
    command = commands.add_parser(
        "train",
        help="Train and evaluate the full SDGF method",
    )
    command.add_argument(
        "--config", required=True, type=Path
    )
    command.add_argument(
        "--data", type=Path,
        help="Override data.path in the config",
    )
    command.add_argument(
        "--output", required=True, type=Path,
        help="New or empty run directory",
    )
    command.add_argument(
        "--epochs", type=positive_int,
        help="Override training.epochs",
    )
    command.add_argument(
        "--batch-size", type=positive_int,
        help="Override training.batch_size",
    )
    command.add_argument(
        "--seed", type=int, help="Override training.seed"
    )


def _inference_parsers(commands):
    """Register evaluation and prediction commands.

    Args:
        commands (SubParsersAction): Command collection.
    """
    descriptions = (
        (
            "evaluate",
            "Re-evaluate the original held-out test split",
        ),
        (
            "predict",
            "Predict for a new chronological table",
        ),
    )
    for name, help_text in descriptions:
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--run-dir", required=True, type=Path
        )
        command.add_argument(
            "--data", required=True, type=Path
        )
        command.add_argument(
            "--output", type=Path,
            required=name == "predict",
        )


def parser():
    """Build the command parser without loading PyTorch.

    Returns:
        ArgumentParser: Complete SDGF command parser.
    """
    result = argparse.ArgumentParser(
        description="SDGF time-series forecasting"
    )
    commands = result.add_subparsers(
        dest="command", required=True
    )
    _prepare_parser(commands)
    _training_parser(commands)
    _inference_parsers(commands)
    for name, command in commands.choices.items():
        if name != "prepare":
            command.add_argument(
                "--device", choices=("auto", "cpu", "cuda"),
                default="auto",
            )
    return result


def _run_training(args):
    """Apply CLI overrides before invoking model training.

    Args:
        args (Namespace): Parsed training arguments.
    """
    from . import engine
    from .config import load_config

    config = load_config(args.config)
    for flag, setting in (
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("seed", "seed"),
    ):
        value = getattr(args, flag)
        if value is not None:
            config["training"][setting] = value
    data_path = args.data or Path(config["data"]["path"])
    engine.train(
        config, data_path, args.output, args.device
    )


def _dispatch(args):
    """Execute one parsed subcommand with lazy imports.

    Args:
        args (Namespace): Parsed command-line options.
    """
    if args.command == "prepare":
        from .preprocessing import prepare_file

        prepare_file(
            args.dataset, args.input, args.output,
            args.date_format, args.encoding,
        )
        return
    from . import engine

    if args.command == "train":
        _run_training(args)
    elif args.command == "evaluate":
        engine.evaluate(
            args.run_dir, args.data, args.output,
            args.device,
        )
    elif args.command == "predict":
        engine.predict(
            args.run_dir, args.data, args.output,
            args.device,
        )


def main(argv=None):
    """Parse arguments and run the selected workflow.

    Args:
        argv (list[str] | None): Explicit arguments or None
            to read arguments from the current process.
    """
    # Set before importing torch for deterministic CUDA.
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
    )
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        _dispatch(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        command_parser.exit(2, f"Error: {exc}\n")

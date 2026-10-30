"""Validate JSON settings and supply training defaults.

See docs/hyperparameters.md for parameter definitions.
"""

import copy
import json
from pathlib import Path

MODEL_DEFAULTS = {
    "dynamic_hidden_dims": [64, 32],
    "dropout": 0.1,
    "use_feature_attention": True,
}
# Benchmark JSON files explicitly contain these settings.
TRAINING_DEFAULTS = {
    "epochs": 500,
    "batch_size": 128,
    # Learning rates follow the model's parameter groups.
    "lr_static": 0.001,
    "lr_main": 0.001,
    "lr_dynamic": 0.0003,
    "weight_decay": 0.00002,
    "patience": 10,
    "min_delta": 0.000001,
    "use_huber": True,
    "use_scheduler": True,
    # The first restart cycle is measured in epochs.
    "scheduler_t0": 30,
    "scheduler_t_mult": 2,
    "eta_min": 0.000001,
    "seed": 42,
    "deterministic": True,
    "num_threads": 1,
}


def _merge_defaults(config, section, defaults):
    """Merge defaults into one configuration section.

    Args:
        config (dict): Configuration to update in place.
        section (str): Section name.
        defaults (dict): Allowed keys and default values.

    Raises:
        ValueError: A section contains an unknown key.
    """
    unknown = set(config.get(section, {})) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown {section} settings: {sorted(unknown)}"
        )
    config[section] = {
        **defaults,
        **config.get(section, {}),
    }


def _validate_features(data):
    """Check feature names and target separation.

    Args:
        data (dict): Data configuration.

    Raises:
        ValueError: Feature lists contain invalid names.
    """
    ts = data["ts_features"]
    static = data["static_features"]
    valid_names = all(
        isinstance(name, str) and name
        for name in ts + static
    )
    if not ts or not valid_names:
        raise ValueError(
            "Provide nonempty time-series feature names "
            "and valid static feature names."
        )
    duplicates = (
        len(set(ts)) != len(ts)
        or len(set(static)) != len(static)
        or set(ts) & set(static)
    )
    if duplicates:
        raise ValueError(
            "Feature names must be unique; time-series "
            "and static lists must be disjoint."
        )
    if data["target"] in static:
        raise ValueError(
            "The current target cannot be a known "
            "static covariate."
        )


def _validate_ratios(data):
    """Check chronological split proportions.

    Args:
        data (dict): Data configuration.

    Raises:
        ValueError: Ratios leave an empty split.
    """
    train_ratio = data["train_ratio"]
    val_ratio = data["val_ratio"]
    if (
        not 0 < train_ratio < 1
        or not 0 < val_ratio < 1
        or train_ratio + val_ratio >= 1
    ):
        raise ValueError(
            "Train and validation ratios must be positive "
            "and sum to less than one."
        )


def _validate_training(training):
    """Check epoch, optimizer and scheduler settings.

    Args:
        training (dict): Resolved training settings.

    Raises:
        ValueError: A setting falls outside its range.
    """
    integer_keys = (
        "epochs", "batch_size", "patience", "num_threads",
        "scheduler_t0", "scheduler_t_mult",
    )
    for key in integer_keys:
        value = training[key]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(
                f"training.{key} "
                "must be a positive integer."
            )
    for key in ("lr_static", "lr_main", "lr_dynamic"):
        if training[key] <= 0:
            raise ValueError(
                f"training.{key} must be positive."
            )
    if any(
        training[key] < 0
        for key in ("weight_decay", "min_delta", "eta_min")
    ):
        raise ValueError(
            "Weight decay, minimum improvement and "
            "minimum LR must be nonnegative."
        )


def _validate_model(model):
    """Check network dimensions and dropout settings.

    Args:
        model (dict): Resolved model settings.

    Raises:
        ValueError: Dropout or hidden sizes are invalid.
    """
    if not 0 <= model["dropout"] < 1:
        raise ValueError("model.dropout must be in [0, 1).")
    dimensions = model["dynamic_hidden_dims"]
    if not dimensions or any(
        not isinstance(dim, int) or dim < 1
        for dim in dimensions
    ):
        raise ValueError(
            "model.dynamic_hidden_dims must contain "
            "positive integers."
        )


def validate_config(config):
    """Return a validated copy with explicit defaults.

    Args:
        config (dict): User-provided configuration.

    Returns:
        dict: Resolved model, data and training sections.

    Raises:
        ValueError: A supplied setting is invalid.
        KeyError: A required data setting is absent.
    """
    config = copy.deepcopy(config)
    _merge_defaults(config, "model", MODEL_DEFAULTS)
    _merge_defaults(config, "training", TRAINING_DEFAULTS)
    data = config["data"]
    data.setdefault("time_column", None)
    data.setdefault("expected_frequency", "1h")
    data.setdefault("train_ratio", 0.7)
    data.setdefault("val_ratio", 0.15)
    _validate_features(data)
    _validate_ratios(data)
    _validate_training(config["training"])
    _validate_model(config["model"])
    return config


def load_config(path):
    """Load and validate a UTF-8 JSON configuration.

    Args:
        path (str | Path): Configuration file path.

    Returns:
        dict: Configuration with resolved defaults.
    """
    with Path(path).open(encoding="utf-8-sig") as handle:
        return validate_config(json.load(handle))

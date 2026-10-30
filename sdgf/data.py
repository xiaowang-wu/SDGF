"""Build chronological windows and training-only scalers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WINDOW = 24


def read_table(path):
    """Read a supported input table.

    Args:
        path (str | Path): UTF-8 CSV or XLSX file path.

    Returns:
        pandas.DataFrame: Loaded table in source row order.

    Raises:
        FileNotFoundError: The source file is absent.
        ValueError: The file extension is unsupported.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}. "
            "See data/README.md."
        )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(
        "Data must be a UTF-8 CSV or an XLSX file."
    )


@dataclass
class Samples:
    """Store windows and their aligned forecast anchors.

    Attributes:
        sequence (ndarray): Inputs shaped [N, F, 24].
        static (ndarray): Known covariates shaped [N, S].
        target (ndarray | None): Labels shaped [N].
        rows (ndarray): Zero-based source row indices.
        times (ndarray | None): Target timestamps.
    """

    sequence: np.ndarray
    static: np.ndarray
    target: np.ndarray | None
    rows: np.ndarray
    times: np.ndarray | None

    def __len__(self):
        """Return the number of forecast samples.

        Returns:
            int: Number of aligned windows.
        """
        return len(self.sequence)


def numeric(frame, columns):
    """Copy selected columns into finite float32 values.

    Args:
        frame (DataFrame): Input rows.
        columns (list[str]): Selected numeric columns.

    Returns:
        ndarray: Writable float32 array.

    Raises:
        ValueError: Values are nonnumeric or nonfinite.
    """
    try:
        values = frame[columns].to_numpy(
            dtype=np.float32, copy=True
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Selected features must be numeric: {columns}"
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(
            "Selected values contain NaN or infinity. "
            "Clean the data without bridging time gaps."
        )
    return values


def _check_columns(frame, config, require_target):
    """Validate columns and the minimum history length.

    Args:
        frame (DataFrame): Source table.
        config (dict): Data configuration.
        require_target (bool): Whether labels are needed.

    Raises:
        ValueError: Required columns or history are missing.
    """
    required = (
        config["ts_features"] + config["static_features"]
    )
    if require_target:
        required = required + [config["target"]]
    if config["time_column"]:
        required = required + [config["time_column"]]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(frame) <= WINDOW:
        raise ValueError(
            f"Need at least {WINDOW + 1} rows "
            "to create one forecast."
        )


def _sample_times(frame, config):
    """Validate timestamps and return forecast-time labels.

    Args:
        frame (DataFrame): Source rows in time order.
        config (dict): Time column and expected interval.

    Returns:
        ndarray | None: Timestamps after the history rows.

    Raises:
        ValueError: Time order or spacing is invalid.
    """
    if not config["time_column"]:
        return None
    times = pd.to_datetime(
        frame[config["time_column"]], errors="raise"
    )
    if (
        times.isna().any()
        or not times.is_monotonic_increasing
        or times.duplicated().any()
    ):
        raise ValueError(
            "Timestamps must be unique, strictly "
            "increasing and nonmissing."
        )
    frequency = config.get("expected_frequency")
    if frequency:
        irregular = (
            times.diff().iloc[1:] != pd.Timedelta(frequency)
        )
        if irregular.any():
            raise ValueError(
                "Timestamps must have a regular "
                f"{frequency} interval; found a gap "
                "or irregular step."
            )
    return times.iloc[WINDOW:].astype(str).to_numpy()


def build_samples(frame, data_config, require_target=True):
    """Construct aligned one-step forecast samples.

    Args:
        frame (DataFrame): Chronological observations.
        data_config (dict): Feature and timestamp settings.
        require_target (bool): Include observed labels.

    Returns:
        Samples: Newest-first windows and target-time data.

    Raises:
        ValueError: Inputs fail schema or time checks.
    """
    _check_columns(frame, data_config, require_target)
    times = _sample_times(frame, data_config)
    history = frame if require_target else frame.iloc[:-1]
    # The final row's time-series values are never inputs.
    # Its target can be absent for next-step prediction.
    values = numeric(history, data_config["ts_features"])
    static = numeric(
        frame.iloc[WINDOW:], data_config["static_features"]
    )
    # Position zero is t-1; exclude the target row.
    sequence = np.stack(
        [
            values[WINDOW - lag:len(frame) - lag]
            for lag in range(1, WINDOW + 1)
        ],
        axis=2,
    )
    target = None
    if require_target:
        target = numeric(
            frame.iloc[WINDOW:], [data_config["target"]]
        ).ravel()
    return Samples(
        sequence, static, target,
        np.arange(WINDOW, len(frame)), times,
    )


def split_bounds(count, data_config):
    """Calculate chronological train/validation boundaries.

    Args:
        count (int): Total number of forecast samples.
        data_config (dict): Train and validation ratios.

    Returns:
        tuple[int, int]: Exclusive split boundaries.

    Raises:
        ValueError: Any split has fewer than two samples.
    """
    train_end = int(count * data_config["train_ratio"])
    val_end = train_end + int(
        count * data_config["val_ratio"]
    )
    sizes = (
        train_end, val_end - train_end, count - val_end
    )
    if min(sizes) < 2:
        raise ValueError(
            "Each chronological split needs at least "
            "two samples; provide more rows."
        )
    return train_end, val_end


def _standardize(values, mean, scale):
    """Apply saved statistics through StandardScaler.

    Args:
        values (ndarray): Two-dimensional feature values.
        mean (ndarray): Training feature means.
        scale (ndarray): Training feature scales.

    Returns:
        ndarray: Standardized values.
    """
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = mean, scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = len(mean)
    # Delegate rounding behavior to the installed sklearn.
    return scaler.transform(values)


class Preprocessor:
    """Store training-only scaler statistics as arrays."""

    def fit(self, samples, train_end):
        """Fit scalers using only training samples.

        Args:
            samples (Samples): All chronological samples.
            train_end (int): Exclusive training boundary.

        Returns:
            Preprocessor: This fitted instance.
        """
        sequence = samples.sequence[:train_end]
        features = sequence.shape[1]
        flat = sequence.transpose(0, 2, 1).reshape(
            -1, features
        )
        # Repeated history values retain original weighting.
        scaler = StandardScaler().fit(flat)
        self.ts_mean, self.ts_scale = (
            scaler.mean_, scaler.scale_
        )
        if samples.static.shape[1]:
            scaler = StandardScaler().fit(
                samples.static[:train_end]
            )
            self.static_mean, self.static_scale = (
                scaler.mean_, scaler.scale_
            )
        else:
            self.static_mean = self.static_scale = (
                np.empty(0)
            )
        return self

    def transform(self, samples):
        """Scale inputs without refitting the statistics.

        Args:
            samples (Samples): Windows to transform.

        Returns:
            tuple[ndarray, ndarray]: Scaled input arrays.
        """
        batch, features, width = samples.sequence.shape
        flat = samples.sequence.transpose(0, 2, 1).reshape(
            -1, features
        )
        sequence = _standardize(
            flat, self.ts_mean, self.ts_scale
        )
        sequence = sequence.reshape(
            batch, width, features
        ).transpose(0, 2, 1)
        static = samples.static.copy()
        if static.shape[1]:
            static = _standardize(
                static, self.static_mean, self.static_scale
            )
        return (
            np.ascontiguousarray(sequence),
            np.ascontiguousarray(static),
        )

    def save(self, path):
        """Save scaler statistics without pickle objects.

        Args:
            path (str | Path): Destination NPZ path.
        """
        np.savez(
            path,
            ts_mean=self.ts_mean,
            ts_scale=self.ts_scale,
            static_mean=self.static_mean,
            static_scale=self.static_scale,
        )

    @classmethod
    def load(cls, path):
        """Restore numeric scaler statistics.

        Args:
            path (str | Path): Saved NPZ path.

        Returns:
            Preprocessor: Instance with restored statistics.
        """
        result = cls()
        keys = (
            "ts_mean", "ts_scale",
            "static_mean", "static_scale",
        )
        with np.load(path, allow_pickle=False) as arrays:
            for key in keys:
                setattr(result, key, arrays[key].copy())
        return result

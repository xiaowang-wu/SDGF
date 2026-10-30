"""Prepare the three benchmark source tables for SDGF.

See data/README.md for sources and processing details.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRESETS = {
    "load_1": {
        "name": "Panama",
        "time": "datetime",
        "target": "nat_demand",
        "ts": [
            f"{variable}_{city}"
            for city in ("toc", "san", "dav")
            for variable in ("T2M", "QV2M", "TQL", "W2M")
        ] + ["nat_demand"],
        "static": [
            "Holiday_ID", "holiday", "school", "hour_sin",
            "hour_cos", "month_sin", "month_cos",
            "is_weekend",
        ],
        "source": (
            "https://www.kaggle.com/datasets/"
            "saurabhshahane/electricity-load-forecasting"
        ),
    },
    "load_2": {
        "name": "Australia",
        "time": "date",
        "target": "pl",
        "ts": ["Dbt", "Dpt", "Wbt", "Hum", "pl"],
        "static": [
            "is_weekend", "hour_sin", "hour_cos",
            "month_sin", "month_cos",
        ],
        "source": (
            "https://ieee-dataport.org/documents/"
            "power-load-forecasting"
        ),
    },
    "load_3": {
        "name": "Dominican Republic",
        "time": "timestamp",
        "target": "active_power_mw",
        "ts": [
            "active_power_mw", "apparent_temperature_c",
            "cloud_cover_percent",
        ],
        "static": [
            "is_holiday", "is_weekend",
            "is_school_vacation",
            "month_sin", "month_cos",
            "hour_sin", "hour_cos",
        ],
        "source": (
            "https://www.kaggle.com/datasets/rafaeluceta/"
            "short-term-forecasting-dataset-seni-"
            "dominican-rep"
        ),
    },
}
DEFAULT_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "ISO8601",
)


def _parse_strings(series, dataset, date_format):
    """Try explicit source date formats in a fixed order.

    Args:
        series (Series): String timestamp values.
        dataset (str): Dataset identifier.
        date_format (str | None): User format override.

    Returns:
        Series: Parsed datetime values.

    Raises:
        ValueError: No accepted date format matches.
    """
    strings = series.astype("string").str.strip()
    formats = list(DEFAULT_DATE_FORMATS)
    if dataset == "load_3":
        # Resolve ambiguous Dominican dates day-first.
        formats = [
            "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M"
        ] + formats
    if date_format:
        formats = [date_format]
    for candidate in formats:
        try:
            return pd.to_datetime(
                strings, format=candidate, errors="raise"
            )
        except ValueError:
            continue
    raise ValueError(
        "Unrecognized timestamp format. "
        "Supply --date-format explicitly."
    )


def parse_timestamps(series, dataset, date_format=None):
    """Parse timestamps without guessing day/month order.

    Args:
        series (Series): Source timestamp column.
        dataset (str): Dataset-specific parsing preset.
        date_format (str | None): Explicit format override.

    Returns:
        Series: Timezone-naive local datetimes.

    Raises:
        ValueError: Dates are missing or timezone-aware.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        result = pd.to_datetime(series, errors="raise")
    else:
        result = _parse_strings(
            series, dataset, date_format
        )
    if result.isna().any():
        raise ValueError("Timestamps cannot be missing.")
    if result.dt.tz is not None:
        raise ValueError(
            "Use the dataset's original "
            "timezone-naive local timestamps."
        )
    return result


def _rename_panama(frame, operations):
    """Resolve supported Panama column aliases.

    Args:
        frame (DataFrame): Source Panama table.
        operations (list): Processing log to update.

    Returns:
        DataFrame: Table with canonical column names.
    """
    aliases = {
        old: new
        for old, new in (
            ("date", "datetime"), ("pl", "nat_demand")
        )
        if old in frame and new not in frame
    }
    if aliases:
        operations.append({"rename_columns": aliases})
        frame = frame.rename(columns=aliases)
    return frame


def _sort_source(frame, dataset, date_format, operations):
    """Parse and order source rows by forecast time.

    Args:
        frame (DataFrame): Mutable source table copy.
        dataset (str): Dataset identifier.
        date_format (str | None): Format override.
        operations (list): Processing log to update.

    Returns:
        DataFrame: Table sorted by parsed timestamps.

    Raises:
        ValueError: The timestamp column is missing.
    """
    column = PRESETS[dataset]["time"]
    if column not in frame:
        raise ValueError(
            f"Missing timestamp column: {column}"
        )
    frame[column] = parse_timestamps(
        frame[column], dataset, date_format
    )
    frame = frame.sort_values(
        column, kind="stable"
    ).reset_index(drop=True)
    operations.extend([
        "parse timestamps with an explicit date convention",
        "sort timestamps ascending",
    ])
    return frame


def _hourly_rows(frame, dataset, operations):
    """Keep hourly observations and validate their spacing.

    Args:
        frame (DataFrame): Time-sorted source table.
        dataset (str): Dataset identifier.
        operations (list): Processing log to update.

    Returns:
        DataFrame: Continuous hourly source observations.

    Raises:
        ValueError: No valid hourly sequence remains.
    """
    column = PRESETS[dataset]["time"]
    if dataset == "load_2":
        # Preserve exact on-the-hour values, not averages.
        selected = frame[column].dt.minute == 0
        frame = frame.loc[selected].copy().reset_index(
            drop=True
        )
        operations.append(
            "retain minute == 0 rows; do not average "
            "half-hourly observations"
        )
    if frame.empty:
        raise ValueError(
            "No observations remain after preparation."
        )
    times = frame[column]
    if times.duplicated().any():
        raise ValueError(
            "Duplicate timestamps found; "
            "no rows were silently discarded."
        )
    if (
        times.diff().iloc[1:] != pd.Timedelta(hours=1)
    ).any():
        raise ValueError(
            "Prepared data has a time gap "
            "or a non-hourly interval."
        )
    return frame


def _add_calendar(frame, dataset, operations):
    """Recompute the original dataset's calendar features.

    Args:
        frame (DataFrame): Mutable hourly table.
        dataset (str): Dataset identifier.
        operations (list): Processing log to update.
    """
    timestamps = frame[PRESETS[dataset]["time"]]
    # Retain each dataset's original encoding precision.
    dtype = (
        np.float32 if dataset == "load_1" else np.float64
    )
    hour = timestamps.dt.hour.to_numpy(dtype=dtype)
    month = timestamps.dt.month.to_numpy(dtype=dtype)
    offset = 1 if dataset == "load_3" else 0
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["month_sin"] = np.sin(
        2 * np.pi * (month - offset) / 12
    )
    frame["month_cos"] = np.cos(
        2 * np.pi * (month - offset) / 12
    )
    operations.append(
        "recompute hour/month sine-cosine encodings; "
        f"month offset {offset}; {dtype.__name__}"
    )
    if dataset in ("load_1", "load_2"):
        frame["is_weekend"] = (
            timestamps.dt.dayofweek >= 5
        ).astype(int)
        operations.append(
            "derive weekend indicator from the timestamp"
        )
    else:
        operations.append(
            "preserve source holiday, weekend "
            "and school-vacation indicators"
        )


def _select_columns(frame, dataset, operations):
    """Keep required columns and reject invalid numbers.

    Args:
        frame (DataFrame): Table with calendar encodings.
        dataset (str): Dataset identifier.
        operations (list): Processing log to update.

    Returns:
        DataFrame: Ordered model input and target columns.

    Raises:
        ValueError: Columns or numeric values are invalid.
    """
    preset = PRESETS[dataset]
    columns = list(dict.fromkeys(
        [preset["time"]] + preset["ts"] + preset["static"]
        + [preset["target"]]
    ))
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Missing required source columns: {missing}"
        )
    selected = frame[columns].copy()
    try:
        values = selected[columns[1:]].to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Selected input columns must be numeric."
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(
            "Selected data contains missing or infinite "
            "values; no imputation is performed."
        )
    operations.append(
        "keep only SDGF input/target columns; "
        "exclude existing predictions and precomputed lags"
    )
    return selected


def _processing_report(frame, dataset, before, operations):
    """Describe the prepared table and performed operations.

    Args:
        frame (DataFrame): Prepared selected table.
        dataset (str): Dataset identifier.
        before (int): Source row count before processing.
        operations (list): Ordered processing log.

    Returns:
        dict: Serializable preparation metadata.
    """
    preset = PRESETS[dataset]
    times = frame[preset["time"]]
    return {
        "dataset": dataset,
        "name": preset["name"],
        "source_url": preset["source"],
        "rows_input": before,
        "rows_output": len(frame),
        "rows_removed": before - len(frame),
        "start": str(times.iloc[0]),
        "end": str(times.iloc[-1]),
        "columns": list(frame.columns),
        "operations": operations,
        "missing_value_imputation": False,
        "outlier_removal": False,
        "normalization": (
            "none here; training fits StandardScaler "
            "on training samples only"
        ),
    }


def prepare_frame(frame, dataset, date_format=None):
    """Apply source-specific processing without scaling.

    Args:
        frame (DataFrame): Downloaded source observations.
        dataset (str): Load 1, 2 or 3 preset identifier.
        date_format (str | None): Timestamp format override.

    Returns:
        tuple[DataFrame, dict]: Prepared data and its log.

    Raises:
        ValueError: Dataset, schema or time checks fail.
    """
    if dataset not in PRESETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    frame = frame.copy()
    before = len(frame)
    operations = []
    if dataset == "load_1":
        frame = _rename_panama(frame, operations)
    frame = _sort_source(
        frame, dataset, date_format, operations
    )
    frame = _hourly_rows(frame, dataset, operations)
    _add_calendar(frame, dataset, operations)
    selected = _select_columns(frame, dataset, operations)
    report = _processing_report(
        selected, dataset, before, operations
    )
    return selected, report


def _check_paths(input_path, output_path):
    """Reject source replacement and existing outputs.

    Args:
        input_path (Path): Existing source file path.
        output_path (Path): New prepared file path.

    Raises:
        ValueError: Output would replace an existing file.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Input and output paths must differ."
        )
    if output_path.exists():
        raise ValueError(
            f"Output already exists: {output_path}. "
            "Choose a new output file."
        )


def _read_source(path, encoding):
    """Read CSV or XLSX before source-specific processing.

    Args:
        path (Path): Source file path.
        encoding (str): Input CSV encoding.

    Returns:
        DataFrame: Loaded source table.

    Raises:
        ValueError: The input extension is unsupported.
    """
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding=encoding)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError("Input must be CSV or XLSX.")


def _write_prepared(frame, path):
    """Save selected prepared observations.

    Args:
        frame (DataFrame): Prepared observations.
        path (Path): Destination CSV or XLSX path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, index=False, engine="openpyxl")


def _save_report(
    report, input_path, output_path, date_format, encoding
):
    """Attach file fingerprints and save the processing log.

    Args:
        report (dict): Processing metadata to update.
        input_path (Path): Source table path.
        output_path (Path): Prepared table path.
        date_format (str | None): Format override.
        encoding (str): Input CSV encoding.

    Returns:
        Path: Saved JSON report path.
    """
    report.update({
        "input_filename": input_path.name,
        "output_filename": output_path.name,
        "input_sha256": hashlib.sha256(
            input_path.read_bytes()
        ).hexdigest(),
        "output_sha256": hashlib.sha256(
            output_path.read_bytes()
        ).hexdigest(),
        "date_format_override": date_format,
        "csv_encoding": encoding,
    })
    path = output_path.with_suffix(
        output_path.suffix + ".preprocess.json"
    )
    content = json.dumps(
        report, ensure_ascii=False, indent=2
    )
    path.write_text(content + "\n", encoding="utf-8")
    return path


def prepare_file(
    dataset, input_path, output_path,
    date_format=None, encoding="utf-8-sig",
):
    """Prepare a local source file and write its report.

    Args:
        dataset (str): Dataset preset identifier.
        input_path (str | Path): Source CSV or XLSX path.
        output_path (str | Path): New prepared table path.
        date_format (str | None): Timestamp format override.
        encoding (str): Input CSV encoding.

    Returns:
        dict: Processing operations and file fingerprints.

    Raises:
        ValueError: A path or input table is invalid.
    """
    input_path, output_path = (
        Path(input_path), Path(output_path)
    )
    _check_paths(input_path, output_path)
    frame = _read_source(input_path, encoding)
    if output_path.suffix.lower() not in (".csv", ".xlsx"):
        raise ValueError("Output must be CSV or XLSX.")
    prepared, report = prepare_frame(
        frame, dataset, date_format
    )
    _write_prepared(prepared, output_path)
    report_path = _save_report(
        report, input_path, output_path,
        date_format, encoding,
    )
    print(
        f"Prepared {dataset}: {report['rows_input']} -> "
        f"{report['rows_output']} rows; saved {output_path}"
    )
    print(f"Processing report: {report_path}")
    return report

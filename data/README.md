# Data sources and preprocessing

The experiments use **Load 1: Panama**, **Load 2: Australia**, and **Load 3: Dominican Republic**. This repository supplies source links and preprocessing code. Download the data from the source providers; neither raw nor processed benchmark files are bundled with the release.

## Download

| ID | Source dataset | Download | Expected CSV | Time column | Target |
| --- | --- | --- | --- | --- | --- |
| Load 1 | Electricity Load Forecasting — Panama | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting) | `continuous_dataset.csv` | `datetime` | `nat_demand` |
| Load 2 | Power Load Forecasting — Australia | [IEEE DataPort](https://ieee-dataport.org/documents/power-load-forecasting) | `Australia.csv` | `date` | `pl` |
| Load 3 | Short-Term Forecasting Dataset SENI Dominican Rep | [Kaggle](https://www.kaggle.com/datasets/rafaeluceta/short-term-forecasting-dataset-seni-dominican-rep) | `datos_modelo.csv` | `timestamp` | `active_power_mw` |

The Australia URL is the source entry recorded with the author's Australia.csv collection. Its source page could not be independently inspected during this update; no unverified DOI or substitute download is supplied. Individual source-link files are included in [sources/](sources/).

Create `data/raw/` in the project root and extract the downloaded CSVs there. CSV and XLSX inputs are supported; `--input` specifies the actual local file path. Preparation reads local data and does not download it.

## Variables used by the method

The full feature order is defined in [load_1.json](../configs/load_1.json), [load_2.json](../configs/load_2.json) and [load_3.json](../configs/load_3.json).

| Dataset | Time-series variables | Known covariates at the forecast time |
| --- | --- | --- |
| Panama | `T2M`, `QV2M`, `TQL`, `W2M` for each of the `_toc`, `_san`, `_dav` source groups, plus `nat_demand`: 13 variables | `Holiday_ID`, `holiday`, `school`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `is_weekend`: 8 covariates |
| Australia | `Dbt`, `Dpt`, `Wbt`, `Hum`, `pl`: 5 variables | `is_weekend`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`: 5 covariates |
| Dominican Republic | `active_power_mw`, `apparent_temperature_c`, `cloud_cover_percent`: 3 variables | `is_holiday`, `is_weekend`, `is_school_vacation`, `month_sin`, `month_cos`, `hour_sin`, `hour_cos`: 7 covariates |

Time-series variables include historical target observations. Only their preceding 24 observations enter a forecast. Known covariates refer to the forecast timestamp and must be available when that forecast is made. The configuration calls them `static_features`; their values can vary over time.

## Dataset-specific processing

### Load 1: Panama

1. Parse `datetime`, sort observations in ascending time order, and retain the supplied hourly measurements.
2. Preserve the source `Holiday_ID`, `holiday` and `school` values.
3. Derive hour, month and weekday from the timestamps. Compute hour/month sine-cosine encodings and set `is_weekend = 1` for Saturday/Sunday.
4. Select the configured 13 time-series variables and 8 known covariates. Previously generated prediction columns are not required.

```text
hour_sin  = sin(2*pi*hour/24)
hour_cos  = cos(2*pi*hour/24)
month_sin = sin(2*pi*month/12)
month_cos = cos(2*pi*month/12)
```

Here `hour` is 0–23 and `month` is 1–12. Calendar arithmetic uses float32 to match the author's processing. The loader also accepts `date -> datetime` and `pl -> nat_demand` aliases when canonical columns are absent. Downloaded versions may differ in stored decimal precision.

### Load 2: Australia

1. Parse year-first `date` values and sort the half-hourly observations.
2. Retain only rows whose timestamp minute equals zero. This keeps the original on-the-hour values; it does not average adjacent half-hourly rows or shift timestamps.
3. Derive weekend and hour/month encodings with the same formulas as Panama, using month numbers 1–12. Calendar arithmetic uses float64.
4. Keep the 5 time-series variables and 5 known covariates listed above.

The author's saved source has 87,648 rows; hourly selection yields 43,824 rows.

### Load 3: Dominican Republic

1. Parse source `timestamp` values as day/month/year, using `%d/%m/%y %H:%M`. For example, `2/1/21 1:00` means **2 January 2021 at 01:00**.
2. Sort correctly parsed timestamps and check hourly continuity.
3. Recompute calendar encodings: hour uses `hour % 24` and month uses `month - 1`, so January starts the annual cycle. Calendar arithmetic uses float64.
4. Preserve supplied `is_holiday`, `is_weekend` and `is_school_vacation` indicators. Keep the configured 3 time-series variables and 7 known covariates.
5. Exclude precomputed target-lag columns; SDGF constructs its own historical windows.

```text
hour_sin  = sin(2*pi*(hour % 24)/24)
hour_cos  = cos(2*pi*(hour % 24)/24)
month_sin = sin(2*pi*(month - 1)/12)
month_cos = cos(2*pi*(month - 1)/12)
```

Already parsed, timezone-naive timestamps in intermediate XLSX files are also accepted. Use `--date-format` when a downloaded file uses a different explicit date format.

## Prepare

After installing the package, run these commands from the project root:

```bash
python -m sdgf prepare --dataset load_1 --input data/raw/continuous_dataset.csv --output data/load_1.csv
python -m sdgf prepare --dataset load_2 --input data/raw/Australia.csv --output data/load_2.csv
python -m sdgf prepare --dataset load_3 --input data/raw/datos_modelo.csv --output data/load_3.csv
```

The generated CSVs match their corresponding configurations. Each output has a `.preprocess.json` report recording its source URL, input/output filenames and SHA-256 hashes, row counts, date range, selected columns and processing operations. Change `--input` if files are stored elsewhere. Existing outputs are not overwritten. The implementation is [sdgf/preprocessing.py](../sdgf/preprocessing.py).

Preparation does not fit scalers, impute missing values, remove outliers or generate predictions from another model. Apart from Australia's specified hourly selection, rows are not silently removed. Missing or infinite selected numeric values, duplicate timestamps and irregular hourly intervals cause an error.

## Model windows, splitting and standardization

For a target at time `t`, the model reads time-series values at `t-1` through `t-24`, ordered newest first, plus known covariates at `t`. The 24 lags form three scales: 1–3, 4–11 and 12–24.

Forecast samples are constructed first, then split chronologically into 70% training, 15% validation and remaining test samples. Boundaries use `int(N * train_ratio)` and `int(N * val_ratio)`. Evaluation is rolling one-step forecasting: earlier observed values can be used by later validation/test forecasts.

A StandardScaler is fit on training window positions for each time-series variable. A separate scaler is fit on training known covariates. Saved statistics are reused for validation, test and prediction. Targets remain in their original units. This pipeline is implemented in [sdgf/data.py](../sdgf/data.py).

## Author-prepared version

These dimensions and dates describe the author's saved processed tables, rather than every upstream version.

| Dataset | Hourly rows | First timestamp | Last timestamp |
| --- | --- | --- | --- |
| Panama | 48,048 | 2015-01-03 01:00 | 2020-06-27 00:00 |
| Australia | 43,824 | 2006-01-01 01:00 | 2011-01-01 00:00 |
| Dominican Republic | 35,039 | 2021-01-02 01:00 | 2024-12-31 23:00 |

| Dataset | Forecast samples | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| Panama | 48,024 | 33,616 | 7,203 | 7,205 |
| Australia | 43,800 | 30,659 | 6,570 | 6,571 |
| Dominican Republic | 35,015 | 24,510 | 5,252 | 5,253 |

Counts follow the actual Python integer split expressions, including floating-point rounding behavior. Source-version or processing changes can alter these counts. Preparation reports and run metadata identify the input file actually used.

## Train

Use the matching configuration, for example:

```bash
python -m sdgf train --config configs/load_1.json --output outputs/load_1
```

See [hyperparameters](../docs/hyperparameters.md) for complete settings, the [main README](../README.md) for the other execution commands, and the [Chinese preprocessing guide](../docs/preprocessing.md) for an additional description of the processing steps.

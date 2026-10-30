# SDGF

Requires Python 3.10+. Run all commands from the project root.

This release provides data preparation, training, evaluation,
and prediction.

Details: [Data sources and processing](data/README.md) · [Hyperparameters](docs/hyperparameters.md).

## Install

```bash
python -m pip install -e .
```

## Prepare data

Download the three datasets using [these links](data/sources/), create `data/raw/`, and extract the source CSV files there. Then run:

```bash
python -m sdgf prepare --dataset load_1 --input data/raw/continuous_dataset.csv --output data/load_1.csv
python -m sdgf prepare --dataset load_2 --input data/raw/Australia.csv --output data/load_2.csv
python -m sdgf prepare --dataset load_3 --input data/raw/datos_modelo.csv --output data/load_3.csv
```

If your files are elsewhere, replace `--input` with the actual path.

## Train

```bash
python -m sdgf train --config configs/load_1.json --output outputs/load_1
python -m sdgf train --config configs/load_2.json --output outputs/load_2
python -m sdgf train --config configs/load_3.json --output outputs/load_3
```

Training saves the model and test results. Use a new output directory for each run. Add `--device cpu` to run on CPU or `--epochs 100` to change the epoch limit.

## Evaluate and predict

Example for Load 1:

```bash
python -m sdgf evaluate --run-dir outputs/load_1 --data data/load_1.csv
python -m sdgf predict --run-dir outputs/load_1 --data data/new_hourly.csv --output outputs/load_1/predictions.csv
```

For other datasets, change the run directory and data path. `predict` requires your own new data file. Use `python -m sdgf --help` for more options.

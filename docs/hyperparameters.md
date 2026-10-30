# Hyperparameters

This document describes the current release settings. The complete model and training settings are written explicitly in [load_1.json](../configs/load_1.json), [load_2.json](../configs/load_2.json) and [load_3.json](../configs/load_3.json). All three use the same settings below; their input columns and target names differ.

Each training run saves its resolved configuration as `outputs/<run>/config.json`. That file includes any command-line overrides and records the settings actually used for the run.

## Data and forecasting protocol

| Setting | Value | Meaning |
| --- | --- | --- |
| Input interval | 1 hour | Checked by `data.expected_frequency = "1h"` |
| Historical window | 24 steps | Fixed by the current implementation |
| Forecast horizon | 1 step | Predict the next hourly target from its preceding history |
| Lag ranges | 1–3, 4–11, 12–24 | Short, mid and long scales, with widths 3, 8 and 13 |
| `data.train_ratio` | 0.70 | First chronological fraction of forecast samples |
| `data.val_ratio` | 0.15 | Next chronological fraction of forecast samples |
| Test fraction | Remaining samples | Approximately 0.15, after integer split boundaries |
| Input standardization | StandardScaler | Fit on training windows and training known covariates only |
| Target standardization | None | Losses and metrics use the original target units |

Window length, scale boundaries and forecast horizon are fixed structural choices, not JSON configuration keys. The sample order is preserved when forming splits. Training batches are shuffled; validation and test predictions retain chronological order.

## Configurable model settings

These keys belong to the `model` section.

| Key | Value | Meaning |
| --- | --- | --- |
| `dynamic_hidden_dims` | `[64, 32]` | Hidden widths of the dynamic logit MLP |
| `dropout` | `0.1` | Dropout used by the dynamic MLP and feature attention |
| `use_feature_attention` | `true` | Apply residual feature attention before the predictor |

The predictor's residual block retains dropout 0.1 independently of `model.dropout`.

## Fixed architecture settings

Let `F` be the number of time-series variables and `S` the number of known covariates.

| Component | Setting |
| --- | --- |
| Context vector | Width `9F + S`: 24-step mean, standard deviation, newest-minus-oldest difference, six most recent values, and known covariates |
| Dynamic network | Linear → ReLU → Dropout for each hidden width, then a linear output of width `24F` |
| Static lag logits | Trainable arrays of shapes `F × 3`, `F × 8`, and `F × 13`, initialized to zero |
| Gate networks | One network per scale: Linear(`9F + S`, 64) → LayerNorm → ReLU → Linear(64, `F`) → Sigmoid |
| Gate sharing | One gate per sample, variable and scale, shared across that scale's lag positions |
| Lag normalization | Softmax over lags within each scale |
| Fused input width | `3F + S` |
| Feature attention | Bottleneck width `max((3F + S) // 4, 8)`; LayerNorm, GELU, dropout and sigmoid; residual reweighting |
| Predictor | Linear(`3F + S`, 128) → residual block(128) → Linear(128, 1) |
| Predictor residual block | Linear(128, 128) → ReLU → Dropout(0.1) → Linear(128, 128), followed by residual addition and ReLU |

These dimensions are implemented in [sdgf/model.py](../sdgf/model.py). They are listed here separately because the configuration loader rejects unknown keys rather than treating them as tunable settings.

## Training settings

These keys belong to the `training` section.

| Key | Value | Meaning |
| --- | --- | --- |
| `epochs` | `500` | Maximum training epochs |
| `batch_size` | `128` | Training and batched inference size |
| `lr_static` | `0.001` | Learning rate for the three static logit arrays |
| `lr_main` | `0.001` | Learning rate for predictor and feature-attention parameters |
| `lr_dynamic` | `0.0003` | Learning rate for the dynamic network and all gate networks |
| `weight_decay` | `0.00002` | Adam weight decay applied to all parameter groups |
| `patience` | `10` | Consecutive validation epochs without sufficient improvement before stopping |
| `min_delta` | `0.000001` | Required absolute improvement in validation RMSE, in target units |
| `use_huber` | `true` | Use PyTorch Smooth L1 loss; `false` selects mean squared error |
| `use_scheduler` | `true` | Enable cosine annealing with warm restarts |
| `scheduler_t0` | `30` | Length of the first learning-rate cycle, in epochs |
| `scheduler_t_mult` | `2` | Multiply each subsequent cycle length by this factor |
| `eta_min` | `0.000001` | Minimum learning-rate value of the cosine schedule |
| `seed` | `42` | Python, NumPy and PyTorch random seed |
| `deterministic` | `true` | Request deterministic PyTorch operations |
| `num_threads` | `1` | CPU compute thread count used by PyTorch |

The optimizer is `torch.optim.Adam`, using its default `betas=(0.9, 0.999)`, `eps=1e-8` and `amsgrad=false`. These optimizer defaults are not exposed as JSON keys. Smooth L1 uses `beta=1.0` and mean reduction. The training DataLoader uses `shuffle=true` and `num_workers=0`.

## Learning-rate schedule and model selection

The scheduler is `CosineAnnealingWarmRestarts`. Each parameter group begins at its own configured learning rate, decreases along its cosine cycle, and returns to that initial rate at a restart. With `scheduler_t0=30` and `scheduler_t_mult=2`, cycle lengths are 30, 60, 120, … epochs; restarts occur after epochs 30, 90, 210, … . The scheduler is stepped once per epoch after validation. Model weights and Adam state are retained at a learning-rate restart.

Early stopping monitors validation RMSE. A checkpoint is accepted when `val_rmse <= best_val_rmse - min_delta`, with the first epoch providing the initial checkpoint. After `patience` consecutive unsuccessful epochs, training stops. The selected model is restored before final test evaluation. Early stopping can end a run before its first learning-rate restart.

## Changing settings

Edit the `model` or `training` section of the corresponding dataset JSON. The CLI can override the epoch limit, batch size and seed directly:

```bash
python -m sdgf train --config configs/load_1.json --output outputs/load_1_seed7 --epochs 100 --batch-size 64 --seed 7
```

Use a new output directory for each run. Relative paths, including the configured input path, are resolved from the current working directory; `--data` can override the configured input file. Device selection is separate from hyperparameters: `--device auto` is the default, and `--device cpu` forces CPU execution.

The optional synthetic configuration [demo.json](../configs/demo.json) uses 30 epochs and batch size 32 for a local trial. It is separate from the three benchmark configurations.

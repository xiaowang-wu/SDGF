"""Train SDGF and load portable forecasting artifacts."""

import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch.nn import functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
)
from torch.utils.data import DataLoader, TensorDataset

from . import __version__
from .config import load_config, validate_config
from .data import (
    Preprocessor,
    build_samples,
    read_table,
    split_bounds,
)
from .model import SDGFusionModel


def write_json(path, value):
    """Write human-readable UTF-8 JSON.

    Args:
        path (str | Path): Destination path.
        value (object): JSON-serializable content.
    """
    content = json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False
    )
    Path(path).write_text(content + "\n", encoding="utf-8")


def sha256(path):
    """Calculate a file fingerprint in bounded chunks.

    Args:
        path (str | Path): File to hash.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def setup(seed, deterministic=True, num_threads=1):
    """Configure random generators and PyTorch execution.

    Args:
        seed (int): Random seed.
        deterministic (bool): Require deterministic kernels.
        num_threads (int): Number of CPU compute threads.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def get_device(name):
    """Resolve an explicit or automatically chosen device.

    Args:
        name (str): One of auto, cpu or cuda.

    Returns:
        torch.device: Selected execution device.

    Raises:
        ValueError: CUDA is requested but unavailable.
    """
    if name == "auto":
        name = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable. Use --device cpu or "
            "install a CUDA-enabled PyTorch build."
        )
    return torch.device(name)


def create_model(config):
    """Construct the full model from resolved settings.

    Args:
        config (dict): Data and model configuration.

    Returns:
        SDGFusionModel: Untrained CPU model.
    """
    data = config["data"]
    return SDGFusionModel(
        len(data["ts_features"]),
        len(data["static_features"]),
        **config["model"],
    )


def optimizer_for(model, cfg):
    """Create Adam with the method's parameter groups.

    Args:
        model (SDGFusionModel): Model to optimize.
        cfg (dict): Resolved training settings.

    Returns:
        torch.optim.Adam: Group-specific learning rates.
    """
    dynamic = list(model.dynamic_net.parameters())
    for module in (
        model.short_gfu, model.mid_gfu, model.long_gfu
    ):
        dynamic.extend(module.parameters())
    groups = [
        {
            "params": [
                model.short_static_logits,
                model.mid_static_logits,
                model.long_static_logits,
            ],
            "lr": cfg["lr_static"],
        },
        {
            "params": model.predictor.parameters(),
            "lr": cfg["lr_main"],
        },
        {
            "params": model.feature_attn.parameters(),
            "lr": cfg["lr_main"],
        },
        {"params": dynamic, "lr": cfg["lr_dynamic"]},
    ]
    return torch.optim.Adam(
        groups, weight_decay=cfg["weight_decay"]
    )


@torch.no_grad()
def infer(model, sequence, static, batch_size, device):
    """Predict batches without tracking gradients.

    Args:
        model (nn.Module): Fitted forecasting model.
        sequence (ndarray): Scaled history [N, F, 24].
        static (ndarray): Scaled known covariates [N, S].
        batch_size (int): Number of samples per batch.
        device (torch.device): Execution device.

    Returns:
        ndarray: Predictions shaped [N].
    """
    model.eval()
    values = []
    for start in range(0, len(sequence), batch_size):
        selection = slice(start, start + batch_size)
        xs = torch.from_numpy(
            sequence[selection]
        ).to(device)
        st = torch.from_numpy(static[selection]).to(device)
        values.append(model(xs, st).cpu().numpy().ravel())
    return np.concatenate(values)


def metrics(y, prediction):
    """Calculate regression metrics in original units.

    Args:
        y (ndarray): Observed targets.
        prediction (ndarray): Model predictions.

    Returns:
        dict: Metrics, sample count and zero-target count.

    Raises:
        ValueError: Predictions contain nonfinite values.
    """
    if not np.isfinite(prediction).all():
        raise ValueError(
            "Model produced nonfinite predictions; "
            "inspect inputs and learning rates."
        )
    # Keep fractional MAPE and expose percent separately.
    mape = float(
        mean_absolute_percentage_error(y, prediction)
    )
    return {
        "r2": float(r2_score(y, prediction)),
        "rmse": float(np.sqrt(
            mean_squared_error(y, prediction)
        )),
        "mae": float(mean_absolute_error(y, prediction)),
        "mape": mape,
        "mape_percent": 100 * mape,
        "n_samples": int(len(y)),
        "n_zero_targets": int(np.count_nonzero(y == 0)),
    }


def prediction_frame(
    samples, prediction, start=0, end=None
):
    """Attach source anchors to prediction values.

    Args:
        samples (Samples): Aligned source samples.
        prediction (ndarray): Predictions for the selection.
        start (int): First selected sample.
        end (int | None): Exclusive selection boundary.

    Returns:
        DataFrame: Anchors, predictions and optional labels.
    """
    selection = slice(start, end)
    frame = pd.DataFrame({
        "row_index": samples.rows[selection],
        "y_pred": prediction,
    })
    if samples.times is not None:
        frame.insert(
            1, "timestamp", samples.times[selection]
        )
    if samples.target is not None:
        frame["y_true"] = samples.target[selection]
    return frame


def _training_loader(sequence, static, target, cfg):
    """Create the shuffled training loader.

    Args:
        sequence (ndarray): Training windows.
        static (ndarray): Training known covariates.
        target (ndarray): Training labels.
        cfg (dict): Batch-size settings.

    Returns:
        DataLoader: Loader using the global seeded RNG.
    """
    dataset = TensorDataset(
        torch.from_numpy(sequence),
        torch.from_numpy(static),
        torch.from_numpy(target[:, None]),
    )
    return DataLoader(
        dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0,
    )


def _scheduler_for(optimizer, cfg):
    """Create the optional cosine restart scheduler.

    Args:
        optimizer (Optimizer): Optimizer to schedule.
        cfg (dict): Scheduler configuration.

    Returns:
        LRScheduler | None: Enabled scheduler or None.
    """
    if not cfg["use_scheduler"]:
        return None
    # T_0 counts epochs; T_mult grows later cycles.
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["scheduler_t0"],
        T_mult=cfg["scheduler_t_mult"],
        eta_min=cfg["eta_min"],
    )


def _package_versions():
    """Collect versions needed to describe a saved run.

    Returns:
        dict[str, str]: Runtime and package versions.
    """
    return {
        "sdgf": __version__,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


class _TrainingRun:
    """Keep one run's inputs, training and output state."""

    def __init__(
        self, config, data_path, output, device_name
    ):
        """Initialize a run without fitting the model.

        Args:
            config (dict): User configuration.
            data_path (str | Path): Prepared source table.
            output (str | Path): New run directory.
            device_name (str): Execution device selector.
        """
        self.config = validate_config(config)
        self.cfg = self.config["training"]
        setup(
            self.cfg["seed"], self.cfg["deterministic"],
            self.cfg["num_threads"],
        )
        self.device = get_device(device_name)
        self.data_path = data_path
        self._prepare_samples()
        self.output = Path(output)
        self._prepare_output()
        self.loader = _training_loader(
            self.sequence[:self.train_end],
            self.static[:self.train_end],
            self.samples.target[:self.train_end], self.cfg,
        )
        self.model = create_model(
            self.config
        ).to(self.device)
        self.optimizer = optimizer_for(self.model, self.cfg)
        self.scheduler = _scheduler_for(
            self.optimizer, self.cfg
        )
        self.loss_fn = (
            F.smooth_l1_loss
            if self.cfg["use_huber"] else F.mse_loss
        )
        self.best_rmse = float("inf")
        self.best_state = None
        self.best_epoch = self.stale = 0
        self.history = []

    def _prepare_samples(self):
        """Build splits and training-only scaled inputs."""
        data = self.config["data"]
        self.samples = build_samples(
            read_table(self.data_path), data
        )
        self.train_end, self.val_end = split_bounds(
            len(self.samples), data
        )
        # Validation/test values never enter scaler fitting.
        self.preprocessing = Preprocessor().fit(
            self.samples, self.train_end
        )
        self.sequence, self.static = (
            self.preprocessing.transform(self.samples)
        )

    def _prepare_output(self):
        """Create a new output directory safely.

        Raises:
            ValueError: The destination is not empty.
        """
        if self.output.exists() and any(
            self.output.iterdir()
        ):
            raise ValueError(
                "Output directory is not empty: "
                f"{self.output}. Choose a new "
                "--output directory."
            )
        self.output.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self):
        """Optimize the model for one complete epoch.

        Returns:
            float: Mean training loss across samples.

        Raises:
            ValueError: Training produces nonfinite loss.
        """
        self.model.train()
        total_loss = 0.0
        for xs, st, y in self.loader:
            self.optimizer.zero_grad()
            prediction = self.model(
                xs.to(self.device), st.to(self.device)
            )
            loss = self.loss_fn(
                prediction, y.to(self.device)
            )
            if not torch.isfinite(loss):
                raise ValueError(
                    "Nonfinite training loss. Check data "
                    "magnitude and learning rates."
                )
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach()) * len(y)
        return total_loss / self.train_end

    def _validation_rmse(self):
        """Evaluate only the validation split for selection.

        Returns:
            float: Validation root mean squared error.
        """
        selection = slice(self.train_end, self.val_end)
        prediction = infer(
            self.model, self.sequence[selection],
            self.static[selection], self.cfg["batch_size"],
            self.device,
        )
        return metrics(
            self.samples.target[selection], prediction
        )["rmse"]

    def _select_checkpoint(self, val_rmse, epoch):
        """Update the best checkpoint and patience counter.

        Args:
            val_rmse (float): Current validation RMSE.
            epoch (int): One-based completed epoch.
        """
        improved = (
            self.best_state is None
            or val_rmse
            <= self.best_rmse - self.cfg["min_delta"]
        )
        if improved:
            self.best_rmse = val_rmse
            self.best_epoch, self.stale = epoch, 0
            self.best_state = {
                name: value.detach().cpu().clone()
                for name, value
                in self.model.state_dict().items()
            }
        else:
            self.stale += 1

    def _fit(self):
        """Fit until the epoch limit or early stopping."""
        for epoch in range(1, self.cfg["epochs"] + 1):
            train_loss = self._train_epoch()
            val_rmse = self._validation_rmse()
            # Step once after validation, as in the source.
            if self.scheduler:
                self.scheduler.step(epoch)
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_rmse,
            })
            print(
                f"Epoch {epoch:03d} | loss "
                f"{train_loss:.6f} | val RMSE "
                f"{val_rmse:.6f}",
                flush=True,
            )
            self._select_checkpoint(val_rmse, epoch)
            if self.stale >= self.cfg["patience"]:
                print(
                    "Early stopping; restoring epoch "
                    f"{self.best_epoch}.", flush=True,
                )
                break

    def _metadata(self):
        """Describe source, split and selected checkpoint.

        Returns:
            dict: Portable run provenance and versions.
        """
        return {
            "format_version": 1,
            "data_sha256": sha256(self.data_path),
            "n_samples": len(self.samples),
            "train_end": self.train_end,
            "val_end": self.val_end,
            "window_size": 24,
            "lag_order": "newest_to_oldest",
            "best_epoch": self.best_epoch,
            "best_val_rmse": self.best_rmse,
            "training_seconds": (
                time.perf_counter() - self.started
            ),
            "device": str(self.device),
            "n_parameters": sum(
                p.numel() for p in self.model.parameters()
            ),
            "versions": _package_versions(),
        }

    def _save(self, result, prediction):
        """Write the selected model and test artifacts.

        Args:
            result (dict): Final test metrics.
            prediction (ndarray): Held-out predictions.
        """
        torch.save(
            self.best_state, self.output / "model.pt"
        )
        self.preprocessing.save(
            self.output / "preprocess.npz"
        )
        write_json(self.output / "config.json", self.config)
        write_json(self.output / "metrics.json", result)
        pd.DataFrame(self.history).to_csv(
            self.output / "history.csv", index=False
        )
        prediction_frame(
            self.samples, prediction, self.val_end
        ).to_csv(
            self.output / "test_predictions.csv",
            index=False,
        )
        write_json(
            self.output / "metadata.json", self._metadata()
        )

    def run(self):
        """Train, restore the selected model and test once.

        Returns:
            dict: Held-out regression metrics.
        """
        self.started = time.perf_counter()
        print(
            f"Device: {self.device}; "
            "samples train/val/test: "
            f"{self.train_end}/"
            f"{self.val_end - self.train_end}/"
            f"{len(self.samples) - self.val_end}",
            flush=True,
        )
        self._fit()
        # Restore validation selection before touching test.
        self.model.load_state_dict(self.best_state)
        prediction = infer(
            self.model, self.sequence[self.val_end:],
            self.static[self.val_end:],
            self.cfg["batch_size"], self.device,
        )
        result = metrics(
            self.samples.target[self.val_end:], prediction
        )
        self._save(result, prediction)
        print(json.dumps(result, indent=2), flush=True)
        return result


def train(config, data_path, output, device_name="auto"):
    """Train the full method and save a portable run.

    Args:
        config (dict): Data, model and training settings.
        data_path (str | Path): Prepared source data.
        output (str | Path): New or empty output directory.
        device_name (str): Auto, cpu or cuda.

    Returns:
        dict: Metrics on the held-out test split.
    """
    return _TrainingRun(
        config, data_path, output, device_name
    ).run()


def load_run(run_dir, device_name="auto"):
    """Restore model, configuration and preprocessing.

    Args:
        run_dir (str | Path): Saved training directory.
        device_name (str): Auto, cpu or cuda.

    Returns:
        tuple: Configuration, model, scaler and device.
    """
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.json")
    cfg = config["training"]
    setup(
        cfg["seed"], cfg["deterministic"],
        cfg["num_threads"],
    )
    device = get_device(device_name)
    model = create_model(config).to(device)
    state = torch.load(
        run_dir / "model.pt",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    preprocessing = Preprocessor.load(
        run_dir / "preprocess.npz"
    )
    return config, model, preprocessing, device


def original_samples(run_dir, data_path, config):
    """Load the exact source used for the saved test split.

    Args:
        run_dir (str | Path): Saved training directory.
        data_path (str | Path): Original prepared file.
        config (dict): Resolved training configuration.

    Returns:
        tuple: Aligned samples and saved run metadata.

    Raises:
        ValueError: The file fingerprint has changed.
    """
    metadata = json.loads(
        (Path(run_dir) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if sha256(data_path) != metadata["data_sha256"]:
        raise ValueError(
            "This is not the original training file "
            "(SHA-256 differs). Use predict for a new "
            "dataset."
        )
    samples = build_samples(
        read_table(data_path), config["data"]
    )
    return samples, metadata


def evaluate(
    run_dir, data_path, output=None, device_name="auto"
):
    """Re-evaluate the original held-out test split.

    Args:
        run_dir (str | Path): Saved training directory.
        data_path (str | Path): Exact original source file.
        output (str | Path | None): Evaluation directory.
        device_name (str): Auto, cpu or cuda.

    Returns:
        dict: Test metrics using restored preprocessing.
    """
    config, model, preprocessing, device = load_run(
        run_dir, device_name
    )
    samples, metadata = original_samples(
        run_dir, data_path, config
    )
    sequence, static = preprocessing.transform(samples)
    start = metadata["val_end"]
    prediction = infer(
        model, sequence[start:], static[start:],
        config["training"]["batch_size"], device,
    )
    result = metrics(samples.target[start:], prediction)
    output = (
        Path(output) if output
        else Path(run_dir) / "evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    prediction_frame(samples, prediction, start).to_csv(
        output / "predictions.csv", index=False
    )
    print(json.dumps(result, indent=2))
    return result


def predict(run_dir, data_path, output, device_name="auto"):
    """Predict every eligible row of a new source table.

    Args:
        run_dir (str | Path): Saved training directory.
        data_path (str | Path): New chronological inputs.
        output (str | Path): Destination prediction CSV.
        device_name (str): Auto, cpu or cuda.

    Raises:
        ValueError: Model predictions are nonfinite.
    """
    config, model, preprocessing, device = load_run(
        run_dir, device_name
    )
    samples = build_samples(
        read_table(data_path), config["data"],
        require_target=False,
    )
    sequence, static = preprocessing.transform(samples)
    prediction = infer(
        model, sequence, static,
        config["training"]["batch_size"], device,
    )
    if not np.isfinite(prediction).all():
        raise ValueError(
            "Model produced nonfinite predictions."
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(samples, prediction).to_csv(
        output, index=False
    )
    print(
        f"Saved {len(prediction)} predictions to {output}"
    )

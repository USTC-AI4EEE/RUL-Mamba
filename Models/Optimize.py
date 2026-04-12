import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import optuna
import optuna.logging
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, QuantileLoss, SMAPE
from torch.utils.data import DataLoader

optuna_logger = logging.getLogger("optuna")


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def build_trainer_kwargs(max_epochs: int, gradient_clip_val: float, logger, callbacks, trainer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    default_kwargs = {
        "max_epochs": max_epochs,
        "gradient_clip_val": gradient_clip_val,
        "callbacks": callbacks,
        "logger": logger,
        "enable_progress_bar": False,
    }
    if torch.cuda.is_available():
        default_kwargs.update({"accelerator": "gpu", "devices": 1})
    else:
        default_kwargs.update({"accelerator": "cpu", "devices": 1})
    default_kwargs.update(trainer_kwargs)
    return default_kwargs


def select_learning_rate(res, learning_rate_range: Tuple[float, float]) -> float:
    losses = np.asarray(res.results.get("loss", []), dtype=float)
    learning_rates = np.asarray(res.results.get("lr", []), dtype=float)
    valid_mask = np.isfinite(losses) & np.isfinite(learning_rates)
    if valid_mask.sum() == 0:
        return float(np.mean(learning_rate_range))
    valid_losses = losses[valid_mask]
    valid_learning_rates = learning_rates[valid_mask]
    best_idx = int(valid_losses.argmin())
    return float(valid_learning_rates[best_idx])


def save_study_artifacts(study: optuna.Study, hpo_dir: Path) -> None:
    hpo_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(hpo_dir / "Study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    best_trial = study.best_trial
    summary = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
    }
    with open(hpo_dir / "Best_Trial.txt", "w", encoding="utf-8") as fout:
        fout.write(json.dumps(summary, ensure_ascii=False, indent=2))


def optimize_hyperparameters(
    Model,
    train_dataloaders: DataLoader,
    val_dataloaders: DataLoader,
    model_path: str,
    max_epochs: int = 100,
    enc_in: int = 11,
    max_encoder_length: int = 24,
    max_prediction_length: int = 1,
    n_trials: int = 100,
    timeout: float = 3600 * 10.0,
    gradient_clip_val_range: Tuple[float, float] = (0.1, 0.5),
    hidden_size_range: Tuple[int, int] = (16, 265),
    hidden_continuous_size_range: Tuple[int, int] = (8, 64),
    attention_head_size_range: Tuple[int, int] = (1, 8),
    n_hidden_layer_range: Tuple[int, int] = (1, 3),
    dropout_range: Tuple[float, float] = (0.05, 0.15),
    learning_rate_range: Tuple[float, float] = (1e-4, 1e-2),
    use_learning_rate_finder: bool = False,
    trainer_kwargs: Dict[str, Any] = {},
    study: optuna.Study = None,
    verbose: Union[int, bool] = None,
    **kwargs,
) -> optuna.Study:
    assert isinstance(train_dataloaders.dataset, TimeSeriesDataSet) and isinstance(
        val_dataloaders.dataset, TimeSeriesDataSet
    ), "dataloaders must be built from timeseriesdataset"

    hpo_dir = Path(model_path)
    trials_dir = hpo_dir / "Trials"
    tb_logs_dir = hpo_dir / "TensorBoard_Logs"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    tb_logs_dir.mkdir(parents=True, exist_ok=True)

    logging_level = {
        None: optuna.logging.get_verbosity(),
        0: optuna.logging.WARNING,
        1: optuna.logging.INFO,
        2: optuna.logging.DEBUG,
    }
    optuna_verbose = logging_level[verbose]
    optuna.logging.set_verbosity(optuna_verbose)

    loss = kwargs.get("loss", QuantileLoss())

    def objective(trial: optuna.Trial) -> float:
        trial_dir = trials_dir / f"Trial_{trial.number}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=str(trial_dir),
            filename="{epoch}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        metrics_callback = MetricsCallback()
        learning_rate_callback = LearningRateMonitor()
        logger = TensorBoardLogger(save_dir=str(tb_logs_dir), name="Optuna", version=f"Trial_{trial.number}")
        gradient_clip_val = trial.suggest_float("gradient_clip_val", *gradient_clip_val_range, log=True)
        callbacks = [
            metrics_callback,
            learning_rate_callback,
            checkpoint_callback,
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ]
        trainer = pl.Trainer(
            **build_trainer_kwargs(
                max_epochs=max_epochs,
                gradient_clip_val=gradient_clip_val,
                logger=logger,
                callbacks=callbacks,
                trainer_kwargs=trainer_kwargs,
            )
        )

        hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
        n_hidden_layer = trial.suggest_int("n_hidden_layer", *n_hidden_layer_range, log=True)
        kwargs["loss"] = copy.deepcopy(loss)

        if Model.__name__ == "FullyConnectedModel":
            model = Model.from_dataset(
                train_dataloaders.dataset,
                input_size=216,
                output_size=1,
                hidden_size=hidden_size,
                n_hidden_layers=n_hidden_layer,
                log_interval=-1,
                loss=MAE(),
            )
        elif "RULMambaVANNetModel" in Model.__name__:
            model = Model.from_dataset(
                train_dataloaders.dataset,
                seq_len=max_encoder_length,
                enc_in=enc_in,
                pred_len=max_prediction_length,
                dropout=trial.suggest_float("dropout", 0.01, 0.2),
                d_model=trial.suggest_int("d_model", 8, 128, step=8),
                n_dec_layer=trial.suggest_int("n_dec_layer", 1, 3, step=1),
                loss=SMAPE(),
            )
        elif "RULMambaNetModel" in Model.__name__ or "RULMambaVSNNetModel" in Model.__name__:
            model = Model.from_dataset(
                train_dataloaders.dataset,
                seq_len=max_encoder_length,
                enc_in=enc_in,
                pred_len=max_prediction_length,
                dropout=trial.suggest_float("dropout", 0.01, 0.2),
                d_model=trial.suggest_int("d_model", 8, 128, step=8),
                n_dec_layer=trial.suggest_int("n_dec_layer", 1, 3, step=1),
                loss=SMAPE(),
            )
        else:
            model = Model.from_dataset(
                train_dataloaders.dataset,
                dropout=trial.suggest_float("dropout", *dropout_range),
                hidden_size=hidden_size,
                hidden_continuous_size=trial.suggest_int(
                    "hidden_continuous_size",
                    hidden_continuous_size_range[0],
                    min(hidden_continuous_size_range[1], hidden_size),
                    log=True,
                ),
                attention_head_size=trial.suggest_int("attention_head_size", *attention_head_size_range),
                log_interval=-1,
                **kwargs,
            )

        if use_learning_rate_finder:
            lr_trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                gradient_clip_val=gradient_clip_val,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            res = lr_trainer.tuner.lr_find(
                model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                early_stop_threshold=10000,
                min_lr=learning_rate_range[0],
                num_training=100,
                max_lr=learning_rate_range[1],
            )
            optimal_lr = select_learning_rate(res, learning_rate_range)
            optuna_logger.info("Using learning rate of %.3g", optimal_lr)
            model.hparams.learning_rate = optimal_lr
        else:
            model.hparams.learning_rate = trial.suggest_float("learning_rate", *learning_rate_range, log=True)

        trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
        return metrics_callback.metrics[-1]["val_loss"].item()

    if study is None:
        study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    save_study_artifacts(study, hpo_dir)
    return study

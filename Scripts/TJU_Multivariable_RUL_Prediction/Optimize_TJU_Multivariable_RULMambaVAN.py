import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import warnings
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESS_DIR = PROJECT_ROOT / "Scripts" / "Data_Process"
for extra_path in (PROJECT_ROOT, DATA_PROCESS_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from Models.Optimize import optimize_hyperparameters
from Models.RULMambaVAN import RULMambaVANNetModel
from TJU_Data_Process import MultiVariateBatteryDataProcess

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

TIME_VARYING_KNOWN_REALS = [
    "voltage mean",
    "voltage std",
    "voltage kurtosis",
    "voltage skewness",
    "CC Q",
    "CC charge time",
    "voltage slope",
    "voltage entropy",
    "current mean",
    "current std",
    "current kurtosis",
    "current skewness",
    "CV Q",
    "CV charge time",
    "current slope",
    "current entropy",
    "Capacity",
]
TIME_VARYING_UNKNOWN_REALS = ["target"]
OUTPUT_ROOT = Path("Outputs/TJU/Multivariable/RULMambaVAN_Optimize")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RULMambaVAN", help="Model name.")
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=64, help="Input sequence length.")
    parser.add_argument("--label-len", dest="label_len", type=int, default=0, help="Start token length.")
    parser.add_argument("--pred-len", dest="pred_len", type=int, default=1, help="Prediction sequence length.")
    parser.add_argument("--data-path", dest="data_path", type=str, default=str(PROJECT_ROOT / "Data" / "TJU_Data" / "Dataset_3_NCM_NCA_Battery_1C.npy"), help="Battery dataset path.")
    parser.add_argument("--rated-capacity", dest="Rated_Capacity", type=float, default=2.5, help="Rated capacity.")
    parser.add_argument("--test-name", dest="test_name", type=str, default="CY25_1", help="Battery data used for test.")
    parser.add_argument("--start-points", dest="start_point_list", type=int, nargs="+", default=[200, 300, 400], help="Start cycles for prediction.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--count", type=int, default=1, help="Number of repeated experiments.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=200, help="Max train epochs.")
    parser.add_argument("--n-trials", dest="n_trials", type=int, default=1, help="Number of optuna trials.")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto", help="Training accelerator.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_log(message: str, log_file, visible: bool = True) -> None:
    if visible:
        print(message)
    log_file.write(f"{message}\n")
    log_file.flush()


def rul_value_error(y_true, y_pred, threshold):
    true_re, pred_re = len(y_true), 0
    for idx in range(len(y_true) - 1):
        if y_true[idx] <= threshold >= y_true[idx + 1]:
            true_re = idx - 1
            break
    for idx in range(len(y_pred) - 1):
        if y_pred[idx] <= threshold:
            pred_re = idx - 1
            break
    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae_error = abs(true_re - pred_re)
    re_score = abs(true_re - pred_re) / true_re if true_re else 1.0
    return rul_real, rul_pred, ae_error, min(re_score, 1.0)


def build_output_dirs(repeat_index: int, start_point: int):
    start_point_dir = OUTPUT_ROOT / f"Repeat_{repeat_index}" / f"Start_Point_{start_point}"
    paths = {
        "repeat_dir": OUTPUT_ROOT / f"Repeat_{repeat_index}",
        "start_point_dir": start_point_dir,
        "base_ckpt_dir": start_point_dir / "Base_Checkpoints",
        "hpo_dir": start_point_dir / "HPO",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_battery_data(data_path: str):
    battery_data = np.load(data_path, allow_pickle=True)
    return battery_data.item()


def build_datasets(df_train, df_test, args):
    mask_len = len(df_train)
    train_frame = df_train[: int(0.8 * mask_len)]
    val_frame = df_train[int(0.8 * mask_len) :]
    dataset_kwargs = dict(
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        min_encoder_length=args.seq_len,
        max_encoder_length=args.seq_len,
        min_prediction_length=args.pred_len,
        max_prediction_length=args.pred_len,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False,
    )
    training = TimeSeriesDataSet(train_frame, **dataset_kwargs)
    validating = TimeSeriesDataSet(val_frame, **dataset_kwargs)
    testing = TimeSeriesDataSet(df_test, **dataset_kwargs)
    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = validating.to_dataloader(train=False, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = testing.to_dataloader(train=False, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    sample_x, (sample_y, _) = next(iter(train_loader))
    feature_stats = {
        "encoder_features": int(sample_x["encoder_cat"].shape[-1] + sample_x["encoder_cont"].shape[-1]),
        "decoder_features": int(sample_x["decoder_cat"].shape[-1] + sample_x["decoder_cont"].shape[-1]),
        "target_shape": list(sample_y[0].shape),
        "train_size": len(train_frame),
        "val_size": len(val_frame),
        "test_size": len(df_test),
        "training_reals": list(training.reals),
        "unknown_reals": list(training.time_varying_unknown_reals),
    }
    return training, train_loader, val_loader, test_loader, feature_stats


def build_trainer(max_epochs: int, default_root_dir: Path, accelerator: str):
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "gradient_clip_val": 0.2,
        "callbacks": [EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10, verbose=False, mode="min")],
        "logger": False,
        "default_root_dir": str(default_root_dir),
    }
    resolved_accelerator = accelerator
    if accelerator == "auto":
        resolved_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer_kwargs.update({"accelerator": resolved_accelerator, "devices": 1})
    return pl.Trainer(**trainer_kwargs)


def evaluate_predictions(y_true, y_pred, rated_capacity):
    mask = y_true >= 0.0
    mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    rmse = float(np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask]))))
    r2 = float(r2_score(y_true[mask], y_pred[mask]))
    rul_real, rul_pred, ae, re = rul_value_error(y_true[mask], y_pred[mask], threshold=rated_capacity * 0.7)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "RUL_Real": int(rul_real),
        "RUL_Pred": int(rul_pred),
        "AE": int(ae),
        "RE": float(re),
    }


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_numpy_outputs(start_point_dir: Path, actual: np.ndarray, prediction: np.ndarray) -> None:
    np.save(start_point_dir / "Actual.npy", actual)
    np.save(start_point_dir / "Prediction.npy", prediction)


def get_best_trial_checkpoint(hpo_dir: Path, trial_number: int) -> Path:
    trial_dir = hpo_dir / "Trials" / f"Trial_{trial_number}"
    ckpt_files = sorted(trial_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {trial_dir}")
    return ckpt_files[0]


def run_base_training(training, train_loader, val_loader, args, paths, feature_stats, log_file):
    model = RULMambaVANNetModel.from_dataset(
        training,
        learning_rate=0.001,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=len(TIME_VARYING_KNOWN_REALS),
        d_model=16,
        n_dec_layer=2,
        dropout=0.1,
        loss=SMAPE(),
    )
    print_log(f"Model name: {args.model}", log_file)
    print_log(f"Train/Val/Test: {feature_stats['train_size']}/{feature_stats['val_size']}/{feature_stats['test_size']}", log_file)
    print_log(f"Input feature num: {len(training.reals)}", log_file)
    print_log(f"Encoder feature num: {feature_stats['encoder_features']}", log_file)
    print_log(f"Decoder feature num: {feature_stats['decoder_features']}", log_file)
    trainer = build_trainer(args.max_epochs, paths["base_ckpt_dir"], args.accelerator)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    base_best_model_path = Path(trainer.checkpoint_callback.best_model_path)
    if base_best_model_path.exists():
        shutil.copy2(base_best_model_path, paths["base_ckpt_dir"] / "Best_Model.ckpt")
    return model, trainer, base_best_model_path


def run_hpo(training, train_loader, val_loader, args, paths):
    study = optimize_hyperparameters(
        RULMambaVANNetModel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        model_path=str(paths["hpo_dir"]),
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        enc_in=len(TIME_VARYING_KNOWN_REALS),
        max_encoder_length=args.seq_len,
        max_prediction_length=args.pred_len,
        gradient_clip_val_range=(0.1, 0.5),
        learning_rate_range=(0.0001, 0.01),
        trainer_kwargs={"limit_train_batches": 128, "accelerator": args.accelerator if args.accelerator != "auto" else ("gpu" if torch.cuda.is_available() else "cpu"), "devices": 1},
        use_learning_rate_finder=False,
    )
    best_trial_ckpt = get_best_trial_checkpoint(paths["hpo_dir"], study.best_trial.number)
    shutil.copy2(best_trial_ckpt, paths["hpo_dir"] / "Best_Model.ckpt")
    return study, best_trial_ckpt


def predict_with_checkpoint(checkpoint_path: Path, test_loader, df_all, start_point: int, args):
    resolved_accelerator = args.accelerator
    if resolved_accelerator == "auto":
        resolved_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if resolved_accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    model = RULMambaVANNetModel.load_from_checkpoint(str(checkpoint_path)).to(device=device)
    predictions = model.predict(test_loader, batch_size=256)
    actuals_df = df_all.loc[df_all["Cycle"] >= start_point, ["Cycle", "target"]]
    actuals = actuals_df["target"].values
    predictions = predictions.detach().cpu().numpy().reshape(-1)
    y_true = actuals * args.Rated_Capacity
    y_pred = predictions * args.Rated_Capacity
    return y_true, y_pred


def run_single_start_point(repeat_index: int, start_point: int, battery_data, args):
    paths = build_output_dirs(repeat_index, start_point)
    log_path = paths["start_point_dir"] / "Train.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        print_log(f"Repeat: {repeat_index}", log_file)
        print_log(f"Start point: {start_point}", log_file)
        df_train, df_test, df_all = MultiVariateBatteryDataProcess(battery_data, args.test_name, start_point, args)
        training, train_loader, val_loader, test_loader, feature_stats = build_datasets(df_train, df_test, args)
        _, trainer, _ = run_base_training(training, train_loader, val_loader, args, paths, feature_stats, log_file)
        print_log(f"Base training epochs: {trainer.current_epoch}", log_file)
        study, best_trial_ckpt = run_hpo(training, train_loader, val_loader, args, paths)
        print_log(f"Best trial: {study.best_trial.number}", log_file)
        print_log(f"Best trial params: {study.best_trial.params}", log_file)
        y_true, y_pred = predict_with_checkpoint(best_trial_ckpt, test_loader, df_all, start_point, args)
        metrics = evaluate_predictions(y_true, y_pred, args.Rated_Capacity)
        save_numpy_outputs(paths["start_point_dir"], y_true, y_pred)
        save_json(paths["start_point_dir"] / "Metrics.json", metrics)
        save_json(
            paths["start_point_dir"] / "Summary.json",
            {
                "repeat": repeat_index,
                "start_point": start_point,
                "feature_stats": feature_stats,
                "best_trial_number": study.best_trial.number,
                "best_trial_params": study.best_trial.params,
                "metrics": metrics,
            },
        )
        print_log(f"Metrics: {metrics}", log_file)
        return {
            "repeat": repeat_index,
            "start_point": start_point,
            "metrics": metrics,
            "prediction": y_pred.tolist(),
            "actual": y_true.tolist(),
        }


def save_repeat_summary(repeat_dir: Path, repeat_results):
    metrics_by_start = {f"Start_Point_{item['start_point']}": item["metrics"] for item in repeat_results}
    save_json(repeat_dir / "Repeat_Summary.json", metrics_by_start)


def save_run_summary(results):
    merged_results = {
        f"Repeat_{item['repeat']}_Start_Point_{item['start_point']}": item["prediction"] for item in results
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(merged_results, OUTPUT_ROOT / "Merged_Results.pth")
    save_json(OUTPUT_ROOT / "Run_Summary.json", results)


def main():
    args = parse_args()
    set_seed(args.seed)
    battery_data = load_battery_data(args.data_path)
    all_results = []
    for repeat_index in range(1, args.count + 1):
        repeat_results = []
        for start_point in args.start_point_list:
            result = run_single_start_point(repeat_index, start_point, battery_data, args)
            repeat_results.append(result)
            all_results.append(result)
        save_repeat_summary(OUTPUT_ROOT / f"Repeat_{repeat_index}", repeat_results)
    save_run_summary(all_results)


if __name__ == "__main__":
    main()

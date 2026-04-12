import argparse
import importlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME_MAP = {
    "autoformer": "Autoformer",
    "fedformer": "FEDformer",
    "mambasimple": "MambaSimple",
    "patchtst": "PatchTST",
    "pathformer": "PathFormer",
    "rulmamba": "RULMamba",
    "timemixer": "TimeMixer",
    "timesnet": "TimesNet",
}


def set_seed(seed, torch_module):
    if seed is None:
        seed = np.random.randint(1e6)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
    return seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "Configs" / "TJU" / "Univariable" / "Base.yaml"),
        help="Base YAML config path.",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name, e.g. autoformer.")
    parser.add_argument("--model-config", type=str, default=None, help="Model override YAML config path.")
    parser.add_argument("--test-name", type=str, default=None, help="Override dataset.test_name.")
    parser.add_argument("--start-points", nargs="+", type=int, default=None, help="Override dataset.start_points.")
    parser.add_argument("--count", type=int, default=None, help="Override train.count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override train.max_epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Override runtime.seed.")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Physical GPU index to use. Sets CUDA_VISIBLE_DEVICES before importing torch.",
    )
    parser.add_argument(
        "--force-cpu-fft",
        action="store_true",
        help="Run FFT-heavy layers on CPU to avoid cuFFT runtime errors on some GPUs.",
    )
    return parser.parse_args()


def setup_runtime_environment():
    cache_root = PROJECT_ROOT / ".cache"
    matplotlib_cache = cache_root / "matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))


def load_training_dependencies():
    setup_runtime_environment()
    try:
        import pytorch_lightning as pl
        import torch
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data.encoders import EncoderNormalizer
        from pytorch_forecasting.metrics import SMAPE
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from sklearn.metrics import r2_score

        from Scripts.Data_Process.TJU_Data_Process import BatteryDataProcess
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Training dependencies are missing. Please install packages from "
            "`RUL-Mamba/requirements.txt` or activate the project runtime environment."
        ) from error

    return SimpleNamespace(
        pl=pl,
        torch=torch,
        TimeSeriesDataSet=TimeSeriesDataSet,
        EncoderNormalizer=EncoderNormalizer,
        SMAPE=SMAPE,
        EarlyStopping=EarlyStopping,
        ModelCheckpoint=ModelCheckpoint,
        r2_score=r2_score,
        set_seed=lambda seed: set_seed(seed, torch),
        BatteryDataProcess=BatteryDataProcess,
    )


def canonicalize_model_name(model_name):
    if model_name is None:
        return None
    return MODEL_NAME_MAP.get(model_name.lower(), model_name)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_model_config_path(args, base_config):
    model_name = canonicalize_model_name(args.model)
    if args.model_config:
        model_config_path = Path(args.model_config)
        if model_name is None:
            model_name = canonicalize_model_name(model_config_path.stem)
    else:
        if model_name is None:
            raise ValueError("Please provide --model or --model-config.")
        model_config_path = (
            PROJECT_ROOT
            / "Configs"
            / base_config["dataset"]["name"]
            / base_config["dataset"]["input_mode"]
            / f"{model_name}.yaml"
        )

    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    return model_name, model_config_path


def apply_cli_overrides(config, args, model_name):
    config["model"]["name"] = model_name
    if args.test_name is not None:
        config["dataset"]["test_name"] = args.test_name
    if args.start_points is not None:
        config["dataset"]["start_points"] = args.start_points
    if args.count is not None:
        config["train"]["count"] = args.count
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        config["train"]["max_epochs"] = args.max_epochs
    if args.seed is not None:
        config["runtime"]["seed"] = args.seed
    if args.gpu_id is not None:
        config["runtime"]["gpu_id"] = args.gpu_id
    return config


def configure_visible_gpu(config):
    gpu_id = config.get("runtime", {}).get("gpu_id")
    if gpu_id is None:
        return None
    if gpu_id < 0:
        raise ValueError("--gpu-id must be greater than or equal to 0.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return gpu_id


def configure_fft_backend(args):
    if args.force_cpu_fft:
        os.environ["RULMAMBA_FORCE_CPU_FFT"] = "1"
        return "cpu"
    return "auto"


def setup_legacy_module_aliases():
    models_pkg = importlib.import_module("Models")
    layers_pkg = importlib.import_module("Models.Layers")
    sys.modules.setdefault("models", models_pkg)
    sys.modules.setdefault("models.layers", layers_pkg)
    sys.modules.setdefault("ModelsModify", models_pkg)
    sys.modules.setdefault("ModelsModify.layers", layers_pkg)
    sys.modules.setdefault("layers", layers_pkg)


def import_object(class_path):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def select_gpu(deps):
    if not deps.torch.cuda.is_available():
        return None
    return 0


def format_output_name(template, config):
    return template.format(
        dataset=config["dataset"]["name"],
        input_mode=config["dataset"]["input_mode"],
        model=config["model"]["name"],
        test_name=config["dataset"]["test_name"],
    )


def ensure_directories(config):
    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    logs_dir = PROJECT_ROOT / config["output"]["logs_dir"]
    outputs_root = (
        PROJECT_ROOT
        / config["output"]["outputs_dir"]
        / config["dataset"]["name"]
        / config["dataset"]["input_mode"]
        / config["model"]["name"]
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    return results_dir, logs_dir, outputs_root


def write_log(message, log_file, visible=True):
    if visible:
        print(message)
    log_file.write(f"{message}\n")
    log_file.flush()


def to_namespace(config):
    return SimpleNamespace(
        Rated_Capacity=config["dataset"]["rated_capacity"],
        seq_len=config["window"]["seq_len"],
    )


def load_battery_data(config):
    cache_path = PROJECT_ROOT / config["dataset"]["battery_cache_path"]
    battery_data = np.load(cache_path, allow_pickle=True)
    return battery_data.item()


def load_real_data(config, torch_module):
    real_data_path = PROJECT_ROOT / config["dataset"]["real_data_path_template"].format(
        test_name=config["dataset"]["test_name"]
    )
    return torch_module.load(real_data_path)


def rul_value_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for index in range(len(y_test) - 1):
        if y_test[index] <= threshold >= y_test[index + 1]:
            true_re = index - 1
            break

    for index in range(len(y_predict) - 1):
        if y_predict[index] <= threshold:
            pred_re = index - 1
            break

    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae_error = abs(true_re - pred_re)
    re_score = abs(true_re - pred_re) / (true_re + 1)
    if re_score > 1:
        re_score = 1
    return rul_real, rul_pred, ae_error, re_score


def build_timeseries_dataset(dataframe, known_reals, unknown_reals, window_config, deps):
    return deps.TimeSeriesDataSet(
        dataframe,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        min_encoder_length=window_config["seq_len"],
        max_encoder_length=window_config["seq_len"],
        min_prediction_length=window_config["pred_len"],
        max_prediction_length=window_config["pred_len"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=deps.EncoderNormalizer(),
        add_encoder_length=False,
    )


def build_dataloaders(df_train, df_test, config, deps):
    train_ratio = config["train"]["train_split_ratio"]
    split_index = int(train_ratio * len(df_train))
    known_reals = config["features"]["time_varying_known_reals"]
    unknown_reals = config["features"]["time_varying_unknown_reals"]

    training = build_timeseries_dataset(
        df_train.iloc[:split_index], known_reals, unknown_reals, config["window"], deps
    )
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=config["train"]["batch_size"],
        shuffle=config["train"]["shuffle_train"],
        num_workers=config["train"]["num_workers"],
        drop_last=config["train"]["drop_last_train"],
    )

    validing = build_timeseries_dataset(
        df_train.iloc[split_index:], known_reals, unknown_reals, config["window"], deps
    )
    val_dataloader = validing.to_dataloader(
        train=False,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        drop_last=False,
    )

    testing = build_timeseries_dataset(df_test, known_reals, unknown_reals, config["window"], deps)
    test_dataloader = testing.to_dataloader(
        train=False,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        drop_last=False,
    )

    batch_x, (batch_y, _) = next(iter(train_dataloader))
    feature_info = {
        "encoder_shape": list(batch_x["encoder_cont"].shape),
        "decoder_shape": list(batch_x["decoder_cont"].shape),
        "target_shape": list(batch_y[0].shape),
        "encoder_feature_count": batch_x["encoder_cat"].shape[-1] + batch_x["encoder_cont"].shape[-1],
        "decoder_feature_count": batch_x["decoder_cat"].shape[-1] + batch_x["decoder_cont"].shape[-1],
    }

    return training, train_dataloader, val_dataloader, test_dataloader, feature_info


def build_common_kwargs(config, enc_in, deps, include_label_len=True, include_dec_in=True):
    build_args = config["model"].get("build_args", {})
    kwargs = {
        "seq_len": build_args.get("seq_len", config["window"]["seq_len"]),
        "pred_len": build_args.get("pred_len", config["window"]["pred_len"]),
        "enc_in": build_args.get("enc_in", enc_in),
        "learning_rate": config["train"]["learning_rate"],
        "loss": deps.SMAPE(),
    }
    if include_label_len:
        kwargs["label_len"] = build_args.get("label_len", config["window"]["label_len"])
    if include_dec_in:
        kwargs["dec_in"] = build_args.get("dec_in", enc_in)
    return kwargs


def build_model_kwargs(config, enc_in, deps):
    model_name = config["model"]["name"].lower()
    build_args = config["model"].get("build_args", {})

    if model_name in {"autoformer", "fedformer"}:
        kwargs = build_common_kwargs(config, enc_in, deps)
    elif model_name == "mambasimple":
        kwargs = {
            "pred_len": build_args.get("pred_len", config["window"]["pred_len"]),
            "enc_in": build_args.get("enc_in", enc_in),
            "c_out": build_args.get("c_out", 1),
            "e_layers": build_args.get("e_layers", 2),
            "d_model": build_args.get("d_model", 16),
            "d_ff": build_args.get("d_ff", 32),
            "expand": build_args.get("expand", 2),
            "d_conv": build_args.get("d_conv", 4),
            "embed": build_args.get("embed", "timeF"),
            "freq": build_args.get("freq", "h"),
            "dropout": build_args.get("dropout", 0.1),
            "learning_rate": config["train"]["learning_rate"],
            "loss": deps.SMAPE(),
        }
    elif model_name == "patchtst":
        kwargs = build_common_kwargs(
            config, enc_in, deps, include_label_len=False, include_dec_in=False
        )
        kwargs["patch_len"] = build_args.get("patch_len", 6)
        kwargs["stride"] = build_args.get("stride", 3)
        kwargs["c_out"] = build_args.get("c_out", 1)
        kwargs["e_layers"] = build_args.get("e_layers", 2)
        kwargs["n_heads"] = build_args.get("n_heads", 8)
        kwargs["factor"] = build_args.get("factor", 3)
        kwargs["d_model"] = build_args.get("d_model", 16)
        kwargs["d_ff"] = build_args.get("d_ff", 32)
        kwargs["dropout"] = build_args.get("dropout", 0.1)
        kwargs["activation"] = build_args.get("activation", "gelu")
        kwargs["output_attention"] = build_args.get("output_attention", False)
    elif model_name == "pathformer":
        kwargs = {
            "seq_len": build_args.get("seq_len", config["window"]["seq_len"]),
            "pred_len": build_args.get("pred_len", config["window"]["pred_len"]),
            "enc_in": build_args.get("enc_in", enc_in),
            "k": build_args["k"],
            "patch_size_list": build_args["patch_size_list"],
            "learning_rate": config["train"]["learning_rate"],
            "loss": deps.SMAPE(),
        }
    elif model_name == "rulmamba":
        kwargs = {
            "seq_len": build_args.get("seq_len", config["window"]["seq_len"]),
            "pred_len": build_args.get("pred_len", config["window"]["pred_len"]),
            "enc_in": build_args.get("enc_in", enc_in),
            "c_out": build_args.get("c_out", 1),
            "d_model": build_args.get("d_model", 16),
            "n_dec_layer": build_args.get("n_dec_layer", 2),
            "dropout": build_args.get("dropout", 0.1),
            "expand": build_args.get("expand", 2),
            "learning_rate": config["train"]["learning_rate"],
            "loss": deps.SMAPE(),
        }
    elif model_name == "timemixer":
        kwargs = build_common_kwargs(
            config, enc_in, deps, include_label_len=False, include_dec_in=False
        )
        kwargs["c_out"] = build_args.get("c_out", 1)
        kwargs["e_layers"] = build_args.get("e_layers", 2)
        kwargs["d_model"] = build_args.get("d_model", 16)
        kwargs["d_ff"] = build_args.get("d_ff", 32)
        kwargs["top_k"] = build_args.get("top_k", 5)
        kwargs["embed"] = build_args.get("embed", "timeF")
        kwargs["freq"] = build_args.get("freq", "h")
        kwargs["dropout"] = build_args.get("dropout", 0.1)
        kwargs["moving_avg"] = build_args.get("moving_avg", 25)
        kwargs["channel_independence"] = build_args.get("channel_independence", 1)
        kwargs["decomp_method"] = build_args.get("decomp_method", "moving_avg")
        kwargs["use_norm"] = build_args.get("use_norm", 1)
        kwargs["down_sampling_layers"] = build_args.get("down_sampling_layers", 3)
        kwargs["down_sampling_window"] = build_args.get("down_sampling_window", 2)
        kwargs["down_sampling_method"] = build_args.get("down_sampling_method", "avg")
    elif model_name == "timesnet":
        kwargs = build_common_kwargs(config, enc_in, deps, include_dec_in=False)
        kwargs["c_out"] = build_args.get("c_out", 1)
        kwargs["e_layers"] = build_args.get("e_layers", 2)
        kwargs["d_model"] = build_args.get("d_model", 16)
        kwargs["d_ff"] = build_args.get("d_ff", 32)
        kwargs["top_k"] = build_args.get("top_k", 5)
        kwargs["embed"] = build_args.get("embed", "timeF")
        kwargs["freq"] = build_args.get("freq", "h")
        kwargs["dropout"] = build_args.get("dropout", 0.1)
        kwargs["num_kernels"] = build_args.get("num_kernels", 6)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return kwargs


def build_trainer(config, start_point_output_dir, selected_gpu, deps):
    checkpoint_callback = deps.ModelCheckpoint(
        dirpath=str(start_point_output_dir / "Checkpoints"),
        filename="{epoch}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_callback = deps.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=config["train"]["patience"],
        verbose=False,
        mode="min",
    )
    trainer_kwargs = {
        "max_epochs": config["train"]["max_epochs"],
        "gradient_clip_val": config["train"]["gradient_clip_val"],
        "callbacks": [early_stop_callback, checkpoint_callback],
        "logger": False,
        "default_root_dir": str(start_point_output_dir),
    }
    if selected_gpu is not None:
        trainer_kwargs["gpus"] = 1
    else:
        trainer_kwargs["gpus"] = 0
    trainer = deps.pl.Trainer(**trainer_kwargs)
    return trainer, checkpoint_callback


def evaluate_predictions(y_true, y_pred, rated_capacity, deps):
    mask = y_true >= 0.0
    nmae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    nrmse = float(np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask]))))
    r2 = float(deps.r2_score(y_true[mask], y_pred[mask]))
    rul_real, rul_pred, ae_error, re_score = rul_value_error(
        y_true[mask], y_pred[mask], threshold=rated_capacity * 0.7
    )
    return {
        "mae": nmae,
        "rmse": nrmse,
        "r2": r2,
        "rul_real": int(rul_real),
        "rul_pred": int(rul_pred),
        "ae": int(ae_error),
        "re": float(re_score),
    }


def json_dump(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def aggregate_start_point_metrics(metrics_list):
    keys = ["mae", "rmse", "r2", "rul_real", "rul_pred", "ae", "re", "train_time", "infer_time", "epoch"]
    summary = {}
    for key in keys:
        values = [metric[key] for metric in metrics_list]
        summary[f"{key}_mean"] = float(np.mean(values))
    return summary


def is_repeat_complete(repeat_dir, start_points):
    metrics_path = repeat_dir / "Metrics.json"
    if not metrics_path.exists():
        return False

    try:
        repeat_metrics = load_yaml(metrics_path)
    except Exception:
        return False

    for start_point in start_points:
        key = f"SP{start_point}"
        if key not in repeat_metrics:
            return False
        prediction_path = repeat_dir / f"Start_Point_{start_point}" / "Prediction.npy"
        if not prediction_path.exists():
            return False
    return True


def load_repeat_outputs(repeat_dir, start_points):
    with open(repeat_dir / "Metrics.json", "r", encoding="utf-8") as file:
        repeat_metrics = json.load(file)

    repeat_predictions = {}
    filtered_metrics = {}
    for start_point in start_points:
        key = f"SP{start_point}"
        prediction_path = repeat_dir / f"Start_Point_{start_point}" / "Prediction.npy"
        repeat_predictions[key] = np.load(prediction_path)
        filtered_metrics[key] = repeat_metrics[key]
    return repeat_predictions, filtered_metrics


def is_start_point_complete(start_point_dir):
    required_files = ["Metrics.json", "Prediction.npy", "Actual.npy"]
    return all((start_point_dir / file_name).exists() for file_name in required_files)


def load_start_point_outputs(start_point_dir):
    with open(start_point_dir / "Metrics.json", "r", encoding="utf-8") as file:
        metrics = json.load(file)
    prediction = np.load(start_point_dir / "Prediction.npy")
    return prediction, metrics


def run_repeat(
    repeat_index,
    battery_data,
    config,
    outputs_root,
    aggregate_log,
    model_class,
    selected_gpu,
    deps,
):
    repeat_seed = deps.set_seed(config["runtime"]["seed"] + repeat_index - 1)
    repeat_dir = outputs_root / f"Repeat_{repeat_index}"
    repeat_dir.mkdir(parents=True, exist_ok=True)

    process_args = to_namespace(config)
    known_reals = config["features"]["time_varying_known_reals"]
    rated_capacity = config["dataset"]["rated_capacity"]
    predict_batch_size = config["train"]["predict_batch_size"]

    repeat_predictions = {}
    repeat_metrics = {}
    for start_point in config["dataset"]["start_points"]:
        start_point_dir = repeat_dir / f"Start_Point_{start_point}"
        start_point_dir.mkdir(parents=True, exist_ok=True)
        key = f"SP{start_point}"

        if is_start_point_complete(start_point_dir):
            write_log(f"resume_skip_start_point={repeat_index}:{start_point}", aggregate_log)
            prediction, metrics = load_start_point_outputs(start_point_dir)
            repeat_predictions[key] = prediction
            repeat_metrics[key] = metrics
            continue

        if start_point_dir.exists() and any(start_point_dir.iterdir()):
            write_log(f"resume_incomplete_start_point={repeat_index}:{start_point}", aggregate_log)

        start_log_path = start_point_dir / "Train.log"
        with open(start_log_path, "w", encoding="utf-8") as repeat_log:
            df_train, df_test, df_all = deps.BatteryDataProcess(
                battery_data,
                config["dataset"]["test_name"],
                start_point,
                process_args,
            )
            training, train_loader, val_loader, test_loader, feature_info = build_dataloaders(
                df_train, df_test, config, deps
            )
            model_kwargs = build_model_kwargs(config, len(known_reals), deps)
            model = model_class.from_dataset(training, **model_kwargs)

            write_log(
                f"[repeat {repeat_index}][SP{start_point}] seed={repeat_seed}",
                repeat_log,
            )
            write_log(
                f"[repeat {repeat_index}][SP{start_point}] train/val/test="
                f"{int(config['train']['train_split_ratio'] * len(df_train))}/"
                f"{len(df_train) - int(config['train']['train_split_ratio'] * len(df_train))}/"
                f"{len(df_test)}",
                repeat_log,
            )
            write_log(
                f"[repeat {repeat_index}][SP{start_point}] encoder_shape={feature_info['encoder_shape']}, "
                f"decoder_shape={feature_info['decoder_shape']}, target_shape={feature_info['target_shape']}",
                repeat_log,
            )
            write_log(
                f"[repeat {repeat_index}][SP{start_point}] params={model.size() / 1e3:.1f}k",
                repeat_log,
            )

            trainer, checkpoint_callback = build_trainer(config, start_point_dir, selected_gpu, deps)
            train_start = time.time()
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            train_time = time.time() - train_start

            best_model_path = checkpoint_callback.best_model_path
            device = deps.torch.device("cuda" if selected_gpu is not None else "cpu")
            best_model = model_class.load_from_checkpoint(best_model_path).to(device=device)

            infer_start = time.time()
            predictions = best_model.predict(test_loader, batch_size=predict_batch_size)
            infer_time = time.time() - infer_start

            predictions = predictions.detach().cpu().numpy().reshape(-1)
            actuals = df_all.loc[df_all["Cycle"] >= start_point, "target"].values
            y_true = actuals * rated_capacity
            y_pred = predictions * rated_capacity

            metrics = evaluate_predictions(y_true, y_pred, rated_capacity, deps)
            metrics["epoch"] = int(trainer.current_epoch)
            metrics["train_time"] = float(train_time)
            metrics["infer_time"] = float(infer_time)
            metrics["best_model_path"] = str(best_model_path)
            metrics["seed"] = int(repeat_seed)

            repeat_predictions[key] = y_pred
            repeat_metrics[key] = metrics

            np.save(start_point_dir / "Prediction.npy", y_pred)
            np.save(start_point_dir / "Actual.npy", y_true)
            json_dump(metrics, start_point_dir / "Metrics.json")

            write_log(
                f"[repeat {repeat_index}][SP{start_point}] "
                f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}, "
                f"RUL_real={metrics['rul_real']}, RUL_pred={metrics['rul_pred']}, "
                f"AE={metrics['ae']}, RE={metrics['re']:.4f}",
                repeat_log,
            )
            write_log(
                f"[repeat {repeat_index}][SP{start_point}] best_model_path={best_model_path}",
                repeat_log,
            )
            write_log(
                f"[repeat {repeat_index}][SP{start_point}] "
                f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}, "
                f"RUL_real={metrics['rul_real']}, RUL_pred={metrics['rul_pred']}, "
                f"AE={metrics['ae']}, RE={metrics['re']:.4f}",
                aggregate_log,
            )

            del best_model
            del model

    deps.torch.save(repeat_predictions, repeat_dir / "Predictions.pth")
    json_dump(repeat_metrics, repeat_dir / "Metrics.json")
    return repeat_predictions, repeat_metrics


def main():
    args = parse_args()
    base_config = load_yaml(args.config)
    model_name, model_config_path = resolve_model_config_path(args, base_config)
    model_override = load_yaml(model_config_path)
    config = deep_merge(base_config, model_override)
    config = apply_cli_overrides(config, args, model_name)
    requested_gpu = configure_visible_gpu(config)
    fft_backend = configure_fft_backend(args)

    deps = load_training_dependencies()
    setup_legacy_module_aliases()
    model_class = import_object(config["model"]["class_path"])
    selected_gpu = select_gpu(deps)
    results_dir, logs_dir, outputs_root = ensure_directories(config)

    merged_config_path = outputs_root / "Merged_Config.yaml"
    save_yaml(config, merged_config_path)

    log_name = format_output_name(config["output"]["log_filename_template"], config)
    log_path = logs_dir / log_name
    result_name = format_output_name(config["output"]["results_filename_template"], config)
    result_path = results_dir / result_name

    battery_data = load_battery_data(config)
    real_data = load_real_data(config, deps.torch)

    aggregated_predictions = {f"SP{sp}": [] for sp in config["dataset"]["start_points"]}
    aggregated_metrics = {f"SP{sp}": [] for sp in config["dataset"]["start_points"]}

    with open(log_path, "w", encoding="utf-8") as aggregate_log:
        write_log(f"dataset={config['dataset']['name']}", aggregate_log)
        write_log(f"input_mode={config['dataset']['input_mode']}", aggregate_log)
        write_log(f"model={config['model']['name']}", aggregate_log)
        write_log(f"test_name={config['dataset']['test_name']}", aggregate_log)
        write_log(f"start_points={config['dataset']['start_points']}", aggregate_log)
        write_log(f"repeat_count={config['train']['count']}", aggregate_log)
        write_log(f"requested_gpu={requested_gpu}", aggregate_log)
        write_log(f"fft_backend={fft_backend}", aggregate_log)
        write_log(f"visible_cuda_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", aggregate_log)
        write_log(f"selected_gpu={selected_gpu}", aggregate_log)
        write_log(f"merged_config={merged_config_path}", aggregate_log)

        for repeat_index in range(1, config["train"]["count"] + 1):
            repeat_dir = outputs_root / f"Repeat_{repeat_index}"
            if is_repeat_complete(repeat_dir, config["dataset"]["start_points"]):
                write_log(f"resume_skip_repeat={repeat_index}", aggregate_log)
                repeat_predictions, repeat_metrics = load_repeat_outputs(
                    repeat_dir, config["dataset"]["start_points"]
                )
            else:
                if repeat_dir.exists():
                    write_log(f"resume_incomplete_repeat={repeat_index}", aggregate_log)
                repeat_predictions, repeat_metrics = run_repeat(
                    repeat_index,
                    battery_data,
                    config,
                    outputs_root,
                    aggregate_log,
                    model_class,
                    selected_gpu,
                    deps,
                )
            for start_point in config["dataset"]["start_points"]:
                key = f"SP{start_point}"
                aggregated_predictions[key].append(repeat_predictions[key])
                aggregated_metrics[key].append(repeat_metrics[key])

        summary = {
            key: aggregate_start_point_metrics(metric_list)
            for key, metric_list in aggregated_metrics.items()
        }
        for start_point in config["dataset"]["start_points"]:
            key = f"SP{start_point}"
            metrics = summary[key]
            write_log(
                f"[summary][{key}] "
                f"MAE={metrics['mae_mean']:.4f}, RMSE={metrics['rmse_mean']:.4f}, "
                f"R2={metrics['r2_mean']:.4f}, RUL_real={metrics['rul_real_mean']:.4f}, "
                f"RUL_pred={metrics['rul_pred_mean']:.4f}, AE={metrics['ae_mean']:.4f}, "
                f"RE={metrics['re_mean']:.4f}",
                aggregate_log,
            )

    results_payload = {
        "dataset": config["dataset"]["name"],
        "input_mode": config["dataset"]["input_mode"],
        "model": config["model"]["name"],
        "test_name": config["dataset"]["test_name"],
        "rated_capacity": config["dataset"]["rated_capacity"],
        "start_points": config["dataset"]["start_points"],
        "repeat_count": config["train"]["count"],
        "real_data_path": config["dataset"]["real_data_path_template"].format(
            test_name=config["dataset"]["test_name"]
        ),
        "predictions": aggregated_predictions,
    }
    deps.torch.save(results_payload, result_path)
    json_dump(
        {
            "config_path": str(merged_config_path),
            "results_path": str(result_path),
            "summary": {
                key: aggregate_start_point_metrics(value)
                for key, value in aggregated_metrics.items()
            },
        },
        outputs_root / "Summary.json",
    )
    np.save(outputs_root / "Real_Data.npy", np.asarray(real_data))
    print(f"Saved aggregated predictions to {result_path}")
    print(f"Saved aggregated log to {log_path}")


if __name__ == "__main__":
    main()

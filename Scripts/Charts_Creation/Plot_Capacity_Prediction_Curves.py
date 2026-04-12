import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
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
    "rulmambavan": "RULMambaVAN",
    "timemixer": "TimeMixer",
    "timesnet": "TimesNet",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Base YAML config path.")
    parser.add_argument("--model", type=str, default=None, help="Model name, e.g. autoformer.")
    parser.add_argument("--model-config", type=str, default=None, help="Model override YAML config path.")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset.name.")
    parser.add_argument("--input-mode", type=str, default=None, help="Override dataset.input_mode.")
    parser.add_argument("--test-name", type=str, default=None, help="Override dataset.test_name.")
    parser.add_argument("--start-points", nargs="+", type=int, default=None, help="Override dataset.start_points.")
    parser.add_argument("--result-path", type=str, default=None, help="Path to aggregated prediction .pth file.")
    parser.add_argument("--real-data-path", type=str, default=None, help="Path to real data .pth file.")
    parser.add_argument("--save-dir", type=str, default=None, help="Custom plot save directory.")
    parser.add_argument(
        "--plot-mode",
        choices=["repeat", "mean", "all"],
        default="all",
        help="Select which figures to generate.",
    )
    return parser.parse_args()


def canonicalize_model_name(model_name):
    if model_name is None:
        return None
    return MODEL_NAME_MAP.get(model_name.lower(), model_name)


def get_display_model_name(model_name):
    return canonicalize_model_name(model_name)


def load_plot_dependencies():
    cache_root = PROJECT_ROOT / ".cache"
    matplotlib_cache = cache_root / "matplotlib"
    fontconfig_cache = cache_root / "fontconfig"
    cache_root.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    fontconfig_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")
    os.environ.setdefault("FONTCONFIG_FILE", "/etc/fonts/fonts.conf")
    os.environ.setdefault("FONTCONFIG_CACHE", str(fontconfig_cache))

    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    matplotlib.use("agg")
    times_font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for font_file in times_font_dir.glob("times*.ttf"):
        font_manager.fontManager.addfont(str(font_file))
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "Times", "serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )
    return mcolors, plt


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
    return model_name, model_config_path


def apply_cli_overrides(config, args, model_name):
    config["model"]["name"] = model_name
    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset
    if args.input_mode is not None:
        config["dataset"]["input_mode"] = args.input_mode
    if args.test_name is not None:
        config["dataset"]["test_name"] = args.test_name
    if args.start_points is not None:
        config["dataset"]["start_points"] = args.start_points
    return config


def format_output_name(template, config):
    return template.format(
        dataset=config["dataset"]["name"],
        input_mode=config["dataset"]["input_mode"],
        model=config["model"]["name"],
        test_name=config["dataset"]["test_name"],
    )


def load_results_payload(path):
    return torch.load(path, map_location="cpu")


def load_real_data(path):
    real_data = torch.load(path, map_location="cpu")
    return np.asarray(real_data)


def ensure_save_dir(config, args):
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = PROJECT_ROOT / config["output"]["plot_dir_template"].format(
            dataset=config["dataset"]["name"],
            input_mode=config["dataset"]["input_mode"],
            test_name=config["dataset"]["test_name"],
            model=config["model"]["name"],
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def draw_repeat_figure(real_data, prediction_dict, start_points, rated_capacity, model_name, save_path, plot_deps):
    mcolors, plt = plot_deps
    colors = list(mcolors.TABLEAU_COLORS.values())
    x = np.arange(1, len(real_data) + 1)
    threshold = np.full(len(real_data), rated_capacity * 0.7)
    display_model_name = get_display_model_name(model_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, real_data, color="tab:red", linewidth=1.5, label="Real data")
    x_is_time_idx = max(start_points) <= len(real_data)
    for index, start_point in enumerate(start_points):
        key = f"SP{start_point}"
        prediction = np.asarray(prediction_dict[key])
        if x_is_time_idx:
            pred_x = x[start_point - 1:start_point - 1 + len(prediction)]
        else:
            pred_x = x[start_point:start_point + len(prediction)]
        ax.plot(
            pred_x,
            prediction,
            color=colors[index % len(colors)],
            linewidth=1.2,
            label=f"SP = {start_point}",
        )
    ax.plot(x, threshold, color="black", linestyle="--", linewidth=1.0, label="Threshold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title(display_model_name)
    ax.legend(prop={"family": "Times New Roman", "size": 14})
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def draw_mean_figure(real_data, predictions, start_points, rated_capacity, model_name, save_path, plot_deps):
    mean_predictions = {}
    for start_point in start_points:
        key = f"SP{start_point}"
        values = predictions[key]
        if isinstance(values, np.ndarray) and values.ndim == 1:
            mean_predictions[key] = values
        else:
            stacked = np.stack([np.asarray(item) for item in values], axis=0)
            mean_predictions[key] = stacked.mean(axis=0)
    draw_repeat_figure(
        real_data, mean_predictions, start_points, rated_capacity, model_name, save_path, plot_deps
    )


def main():
    args = parse_args()
    base_config = load_yaml(args.config)
    model_name, model_config_path = resolve_model_config_path(args, base_config)
    config = deep_merge(base_config, load_yaml(model_config_path))
    config = apply_cli_overrides(config, args, model_name)

    if args.result_path is not None:
        result_path = Path(args.result_path)
    else:
        result_name = format_output_name(config["output"]["results_filename_template"], config)
        result_path = PROJECT_ROOT / config["output"]["results_dir"] / result_name

    if args.real_data_path is not None:
        real_data_path = Path(args.real_data_path)
    else:
        real_data_path = PROJECT_ROOT / config["dataset"]["real_data_path_template"].format(
            test_name=config["dataset"]["test_name"]
        )

    save_dir = ensure_save_dir(config, args)
    payload = load_results_payload(result_path)
    real_data = load_real_data(real_data_path)
    plot_deps = load_plot_dependencies()

    start_points = payload.get("start_points", config["dataset"]["start_points"])
    predictions = payload["predictions"] if "predictions" in payload else payload
    rated_capacity = payload.get("rated_capacity", config["dataset"]["rated_capacity"])
    first_value = predictions[f"SP{start_points[0]}"]
    repeat_count = payload.get("repeat_count")
    if repeat_count is None:
        if isinstance(first_value, np.ndarray) and first_value.ndim == 1:
            repeat_count = 1
        elif isinstance(first_value, list) and first_value and not isinstance(first_value[0], (list, np.ndarray)):
            repeat_count = 1
        else:
            repeat_count = len(first_value)

    if args.plot_mode in {"repeat", "all"}:
        for repeat_index in range(1, repeat_count + 1):
            repeat_prediction_dict = {}
            for start_point in start_points:
                value = predictions[f"SP{start_point}"]
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    repeat_prediction_dict[f"SP{start_point}"] = value
                elif isinstance(value, list) and value and not isinstance(value[0], (list, np.ndarray)):
                    repeat_prediction_dict[f"SP{start_point}"] = np.asarray(value)
                else:
                    repeat_prediction_dict[f"SP{start_point}"] = value[repeat_index - 1]
            draw_repeat_figure(
                real_data,
                repeat_prediction_dict,
                start_points,
                rated_capacity,
                config["model"]["name"],
                save_dir / f"Repeat_{repeat_index}_Capacity_Prediction_Curves",
                plot_deps,
            )

    if args.plot_mode in {"mean", "all"}:
        draw_mean_figure(
            real_data,
            predictions,
            start_points,
            rated_capacity,
            config["model"]["name"],
            save_dir / "Mean_Capacity_Prediction_Curves",
            plot_deps,
        )
    print(f"Saved plots to {save_dir}")


if __name__ == "__main__":
    main()

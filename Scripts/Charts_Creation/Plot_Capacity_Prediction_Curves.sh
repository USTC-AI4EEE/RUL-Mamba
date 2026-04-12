#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/data/huangjh/anaconda/envs/rulmamba/bin/python"
CONFIG_PATH="${PROJECT_ROOT}/Configs/TJU/Univariable/Base.yaml"
TEST_NAME="CY25_1"
PLOT_MODE="all"

print_usage() {
  cat <<EOF
Usage:
  $(basename "$0") --config PATH --model MODEL [options]

Options:
  --config PATH         Base config file.
  --model MODEL         Model name.
  --model-config PATH   Model config file. Defaults to Configs/<dataset>/<input_mode>/<model>.yaml inferred from config.
  --dataset NAME        Override dataset name.
  --input-mode MODE     Override input mode.
  --test-name NAME      Override test battery name. Default: ${TEST_NAME}
  --plot-mode MODE      repeat | mean | all. Default: ${PLOT_MODE}
  --python-bin PATH     Python interpreter. Default: ${PYTHON_BIN}
  --result-path PATH    Manually provide prediction result file.
  --real-data-path PATH Manually provide real data file.
  --save-dir PATH       Manually provide output plot directory.
  --start-points ...    Override start points.
  -h, --help            Show this help.

Examples:
  $(basename "$0") --config "${PROJECT_ROOT}/Configs/TJU/Univariable/Base.yaml" --model RULMamba --plot-mode mean
  $(basename "$0") --config "${PROJECT_ROOT}/Configs/NASA/Univariable/Base.yaml" --model PatchTST --test-name B0005
EOF
}

MODEL=""
MODEL_CONFIG=""
DATASET_NAME=""
INPUT_MODE=""
FORWARDED_ARGS=()

infer_model_config() {
  local config_path="$1"
  local model_name="$2"
  local dataset_name=""
  local input_mode=""
  local candidate_path=""

  dataset_name=$(python3 - <<'PY' "$config_path"
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
print(data.get('dataset', {}).get('name', ''))
PY
)

  input_mode=$(python3 - <<'PY' "$config_path"
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
print(data.get('dataset', {}).get('input_mode', ''))
PY
)

  if [[ -z "$dataset_name" || -z "$input_mode" ]]; then
    echo "Failed to infer dataset.name or dataset.input_mode from ${config_path}" >&2
    exit 1
  fi

  DATASET_NAME="$dataset_name"
  INPUT_MODE="$input_mode"
  candidate_path="${PROJECT_ROOT}/Configs/${dataset_name}/${input_mode}/${model_name}.yaml"
  if [[ -f "$candidate_path" ]]; then
    printf '%s\n' "$candidate_path"
    return 0
  fi

  candidate_path="${PROJECT_ROOT}/Configs/${dataset_name}/${model_name}.yaml"
  if [[ -f "$candidate_path" ]]; then
    printf '%s\n' "$candidate_path"
    return 0
  fi

  echo "Failed to infer model config path for ${dataset_name}/${input_mode}/${model_name}" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model-config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --test-name)
      TEST_NAME="$2"
      FORWARDED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --plot-mode)
      PLOT_MODE="$2"
      FORWARDED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dataset|--input-mode|--result-path|--real-data-path|--save-dir)
      FORWARDED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --start-points)
      FORWARDED_ARGS+=("$1")
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        FORWARDED_ARGS+=("$1")
        shift
      done
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  print_usage >&2
  exit 1
fi

if [[ -z "$MODEL_CONFIG" ]]; then
  MODEL_CONFIG="$(infer_model_config "$CONFIG_PATH" "$MODEL")"
fi

echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG_PATH}"
echo "Model: ${MODEL}"
echo "Model config: ${MODEL_CONFIG}"
if [[ -n "$DATASET_NAME" && -n "$INPUT_MODE" ]]; then
  echo "Inferred dataset/input mode: ${DATASET_NAME}/${INPUT_MODE}"
fi
echo "Test name: ${TEST_NAME}"
echo "Plot mode: ${PLOT_MODE}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/Scripts/Charts_Creation/Plot_Capacity_Prediction_Curves.py" \
  --config "${CONFIG_PATH}" \
  --model "${MODEL}" \
  --model-config "${MODEL_CONFIG}" \
  --test-name "${TEST_NAME}" \
  --plot-mode "${PLOT_MODE}" \
  "${FORWARDED_ARGS[@]}"

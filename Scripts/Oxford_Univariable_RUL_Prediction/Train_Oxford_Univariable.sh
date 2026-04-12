#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/data/huangjh/anaconda/envs/rulmamba/bin/python"
BASE_CONFIG="${PROJECT_ROOT}/Configs/Oxford/Univariable/Base.yaml"
TEST_NAME="Cell8"

ALL_MODELS=(
  "Autoformer"
  "FEDformer"
  "MambaSimple"
  "PatchTST"
  "PathFormer"
  "RULMamba"
  "TimeMixer"
  "TimesNet"
)

contains_model() {
  local candidate="$1"
  shift
  local model
  for model in "$@"; do
    if [[ "$model" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

print_usage() {
  cat <<EOF
Usage:
  $(basename "$0") [all|MODEL ...] [--test-name NAME] [--python-bin PATH] [--base-config PATH] [-- <train-script-args>]

Examples:
  $(basename "$0")
  $(basename "$0") all --count 10 --gpu-id 0
  $(basename "$0") Autoformer PatchTST --test-name Cell8 -- --max-epochs 100
EOF
}

SELECTED_MODELS=()
FORWARDED_ARGS=()
PARSING_MODELS=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-name)
      [[ $# -ge 2 ]] || { echo "Missing value for --test-name" >&2; exit 1; }
      TEST_NAME="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || { echo "Missing value for --python-bin" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --base-config)
      [[ $# -ge 2 ]] || { echo "Missing value for --base-config" >&2; exit 1; }
      BASE_CONFIG="$2"
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    --)
      shift
      FORWARDED_ARGS+=("$@")
      break
      ;;
    --*)
      FORWARDED_ARGS+=("$1")
      PARSING_MODELS=false
      shift
      ;;
    *)
      if $PARSING_MODELS; then
        if [[ "$1" == "all" ]]; then
          SELECTED_MODELS=("${ALL_MODELS[@]}")
          PARSING_MODELS=false
          shift
          continue
        fi

        if contains_model "$1" "${ALL_MODELS[@]}"; then
          SELECTED_MODELS+=("$1")
          shift
          continue
        fi

        echo "Unknown model: $1" >&2
        echo "Available models: ${ALL_MODELS[*]} or all" >&2
        exit 1
      fi

      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
  SELECTED_MODELS=("${ALL_MODELS[@]}")
fi

echo "Python: ${PYTHON_BIN}"
echo "Base config: ${BASE_CONFIG}"
echo "Test name: ${TEST_NAME}"
echo "Selected models: ${SELECTED_MODELS[*]}"

for MODEL in "${SELECTED_MODELS[@]}"; do
  MODEL_CONFIG="${PROJECT_ROOT}/Configs/Oxford/Univariable/${MODEL}.yaml"
  echo "===== Training ${MODEL} ====="
  "${PYTHON_BIN}" "${PROJECT_ROOT}/Scripts/Oxford_Univariable_RUL_Prediction/Train_Oxford_Univariable.py" \
    --config "${BASE_CONFIG}" \
    --model "${MODEL}" \
    --model-config "${MODEL_CONFIG}" \
    --test-name "${TEST_NAME}" \
    "${FORWARDED_ARGS[@]}"
done

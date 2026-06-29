#!/usr/bin/env bash
# Wrapper to run `run_training.py` with the same CLI options from a shell.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_training.py"

if [ ! -f "$PY_SCRIPT" ]; then
  echo "Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

# Try to activate a local virtualenv if present (POSIX path)
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  . "$SCRIPT_DIR/.venv/bin/activate"
fi

if [ "$#" -eq 0 ]; then
  echo "No args provided. Showing help from run_training.py:"
  python "$PY_SCRIPT" --help
  exit 0
fi

# Parse a simple set of helper arguments for multiple-trial runs.
# Supported modes:
# - --config path.json   (requires `jq` to parse JSON array of trial objects)
# - or provide comma-separated lists for --datasets, --recognition-types, --modalities, --models, --use-pca-options

# Defaults
DELAY=1
LOG_PATH=""
CONFIG_PATH=""
DATASETS=""
RECOG_TYPES=""
MODALITIES=""
MODELS=""
USE_PCA_OPTS=""
FOLDS=5
SAVE_FOLDER_APP=""
PLOT_GLOBAL="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"; shift 2;;
    --datasets)
      DATASETS="$2"; shift 2;;
    --recognition-types)
      RECOG_TYPES="$2"; shift 2;;
    --modalities)
      MODALITIES="$2"; shift 2;;
    --models)
      MODELS="$2"; shift 2;;
    --use-pca-options)
      USE_PCA_OPTS="$2"; shift 2;;
    --folds)
      FOLDS="$2"; shift 2;;
    --delay)
      DELAY="$2"; shift 2;;
    --save-folder-app)
      SAVE_FOLDER_APP="$2"; shift 2;;
    --plot)
      PLOT_GLOBAL="true"; shift 1;;
    --no-plot)
      PLOT_GLOBAL="false"; shift 1;;
    --log)
      LOG_PATH="$2"; shift 2;;
    --) shift; break;;
    *)
      # Unknown or positional arg: forward to python by collecting remaining args
      break
      ;;
  esac
done

# Activate venv if present
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  . "$SCRIPT_DIR/.venv/bin/activate"
fi

run_one() {
  # $1 is a JSON-like string of options or an array of positional args
  local cmd=(python "$PY_SCRIPT")
  local dataset="$1"
  local recognition_type="$2"
  local modality="$3"
  local use_pca_flag="$4"
  local model_type="$5"
  local folds="$6"
  local save_folder_app="$7"
  local plot_flag="$8"

  cmd+=(--dataset "$dataset")
  if [ -n "$recognition_type" ]; then
    cmd+=(--recognition-type "$recognition_type")
  fi
  cmd+=(--modality "$modality")
  if [ "$use_pca_flag" = "true" ]; then
    cmd+=(--use-pca)
  fi
  cmd+=(--model-type "$model_type")
  cmd+=(--folds "$folds")
  if [ -n "$save_folder_app" ]; then
    cmd+=(--save-folder-app "$save_folder_app")
  fi
  # run_training.py uses --no-plot to disable plotting; default is plotting enabled
  if [ "$plot_flag" = "false" ]; then
    cmd+=(--no-plot)
  fi

  echo
  echo "Running: ${cmd[*]}"
  if [ -n "$LOG_PATH" ]; then
    { echo "--- TRIAL: $(date -Iseconds) ---"; echo "${cmd[*]}"; } >> "$LOG_PATH"
  fi
  "${cmd[@]}"
  local status=$?
  echo "Trial exit code: $status"
  if [ -n "$LOG_PATH" ]; then
    echo "exit_code: $status" >> "$LOG_PATH"
  fi
}

# Helper to split comma lists into arrays
split_csv() {
  IFS=',' read -r -a arr <<< "$1"
  # Trim whitespace
  for i in "${!arr[@]}"; do
    arr[$i]="$(echo "${arr[$i]}" | sed -e 's/^\s*//' -e 's/\s*$//')"
  done
  echo "${arr[@]}"
}

if [ -n "$CONFIG_PATH" ]; then
  if ! command -v jq >/dev/null 2>&1; then
    echo "Error: parsing JSON config requires 'jq' to be installed." >&2
    exit 1
  fi
  if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
  fi

  len=$(jq 'length' "$CONFIG_PATH")
  for i in $(seq 0 $((len-1))); do
    dataset=$(jq -r ".[$i].dataset // \"text&soft\"" "$CONFIG_PATH")
    recognition_type=$(jq -r ".[$i].recognition_type // empty" "$CONFIG_PATH")
    modality=$(jq -r ".[$i].modality // \"press\"" "$CONFIG_PATH")
    use_pca=$(jq -r ".[$i].use_pca // false" "$CONFIG_PATH")
    model_type=$(jq -r ".[$i].model_type // \"CNN-LSTM\"" "$CONFIG_PATH")
    folds=$(jq -r ".[$i].folds // $FOLDS" "$CONFIG_PATH")
    plot_results=$(jq -r ".[$i].plot_results // null" "$CONFIG_PATH")
    if [ "$plot_results" = "null" ]; then
      plot_results="${PLOT_GLOBAL}"
    fi
    save_folder_app=$(jq -r ".[$i].save_folder_app // \"\"" "$CONFIG_PATH")
    # Prefer per-trial save_folder_app, fallback to global CLI option
    if [ -z "$save_folder_app" ]; then
      save_folder_app="$SAVE_FOLDER_APP"
    fi
    run_one "$dataset" "$recognition_type" "$modality" "$use_pca" "$model_type" "$folds" "$save_folder_app" "$plot_results"
    sleep "$DELAY"
  done
  exit 0
fi

# If lists provided, build Cartesian product
if [ -n "$DATASETS" ] || [ -n "$MODALITIES" ] || [ -n "$MODELS" ]; then
  datasets_arr=( $(split_csv "${DATASETS:-text&soft}") )
  recog_arr=( $(split_csv "${RECOG_TYPES:-}") )
  modalities_arr=( $(split_csv "${MODALITIES:-press}") )
  models_arr=( $(split_csv "${MODELS:-CNN-LSTM}") )
  usepca_arr=( $(split_csv "${USE_PCA_OPTS:-false}") )

  # If recognition types empty, use empty value to allow defaulting
  if [ ${#recog_arr[@]} -eq 0 ]; then
    recog_arr=("")
  fi

  for ds in "${datasets_arr[@]}"; do
    for rt in "${recog_arr[@]}"; do
      for md in "${modalities_arr[@]}"; do
        for mt in "${models_arr[@]}"; do
          for up in "${usepca_arr[@]}"; do
            # normalize boolean string
            up_norm=$(echo "$up" | awk '{print tolower($0)}')
            if [ "$md" != "all" ]; then
              up_norm=false
            fi
            run_one "$ds" "$rt" "$md" "$up_norm" "$mt" "$FOLDS" "$SAVE_FOLDER_APP" "$PLOT_GLOBAL"
            sleep "$DELAY"
          done
        done
      done
    done
  done
  exit 0
fi

# Default fallback: forward remaining args to run_training.py
echo "Forwarding args to run_training.py: $@"
python "$PY_SCRIPT" "$@"

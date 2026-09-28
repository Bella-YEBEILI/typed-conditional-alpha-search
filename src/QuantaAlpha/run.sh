#!/bin/bash
# QuantaAlpha main experiment runner
#
# Usage:
#   bash run.sh "initial direction" "library suffix" "data domain"
#   CONFIG=configs/experiment.yaml ./run.sh "direction"
#
# Examples:
#   bash run.sh "price-volume factor mining" "pv_1" "pv"
#   bash run.sh "intraday momentum" "min_1" "minutes"
#   bash run.sh "joint pv/minute idea" "joint_1" "pv/minutes"

normalize_factor_data_domains() {
    local domains_raw="$1"
    local tokens
    local token
    local normalized=()
    local joined=""

    tokens="$(printf '%s' "${domains_raw}" | tr '[:upper:]' '[:lower:]' | tr '/,+' '   ')"
    for token in ${tokens}; do
        case "${token}" in
            ""|auto|default|follow_mode|follow-mode)
                continue
                ;;
            pv|daily|day|price_volume|price-volume|pricevolume)
                token="pv"
                ;;
            fundamental|fundamentals|fund|funda)
                token="fundamental"
                ;;
            minutes|minute|intraday|min|1m)
                token="minutes"
                ;;
            combined|all)
                normalized=("pv" "fundamental" "minutes")
                break
                ;;
            *)
                echo "Error: invalid data domain '${token}'. Expected pv, fundamental, minutes, or a combination like pv/minutes." >&2
                return 1
                ;;
        esac

        if [[ " ${normalized[*]} " != *" ${token} "* ]]; then
            normalized+=("${token}")
        fi
    done

    if [ "${#normalized[@]}" -eq 0 ]; then
        return 0
    fi

    for token in "${normalized[@]}"; do
        if [ -z "${joined}" ]; then
            joined="${token}"
        else
            joined="${joined}/${token}"
        fi
    done
    echo "${joined}"
}

infer_factor_data_mode() {
    local domains_raw="$1"
    local domains
    domains="$(normalize_factor_data_domains "${domains_raw}")" || return 1

    if [ -z "${domains}" ]; then
        return 0
    fi

    case "/${domains}/" in
        */minutes/*)
            echo "minutes"
            return 0
            ;;
        */fundamental/*)
            echo "fundamental"
            return 0
            ;;
        */pv/*)
            echo "daily"
            return 0
            ;;
        *)
            return 0
            ;;
    esac
}

normalize_factor_data_mode() {
    local mode_raw="$1"
    local mode
    mode="$(echo "${mode_raw}" | tr '[:upper:]' '[:lower:]')"

    case "${mode}" in
        daily)
            echo "daily"
            return 0
            ;;
        fundamental|fundamentals|fund|funda)
            echo "fundamental"
            return 0
            ;;
        minutes|minute|intraday|min|1m)
            echo "minutes"
            return 0
            ;;
        "")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# =============================================================================
# Locate project root
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# =============================================================================
# Load .env configuration
# =============================================================================
load_env_file() {
    local env_file="$1"
    local line
    local line_no=0
    local trimmed

    while IFS= read -r line || [ -n "${line}" ]; do
        line_no=$((line_no + 1))
        line="${line%$'\r'}"
        if [ "${line_no}" -eq 1 ]; then
            line="${line#$'\xEF\xBB\xBF'}"
        fi

        trimmed="${line#"${line%%[![:space:]]*}"}"
        if [ -z "${trimmed}" ]; then
            continue
        fi
        case "${trimmed}" in
            \#*)
                continue
                ;;
        esac
        if [[ ! "${trimmed}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            echo "Warning: skipping invalid .env line ${line_no}: ${trimmed}"
            continue
        fi

        eval "export ${trimmed}"
    done < "${env_file}"
}

if [ -f "${SCRIPT_DIR}/.env" ]; then
    load_env_file "${SCRIPT_DIR}/.env"
else
    echo "Error: .env file not found"
    echo "Please create ${SCRIPT_DIR}/.env before running this script."
    exit 1
fi

export QUANTAALPHA_QUIET_CONSOLE="${QUANTAALPHA_QUIET_CONSOLE:-1}"
export QUANTAALPHA_DISABLE_TQDM="${QUANTAALPHA_DISABLE_TQDM:-1}"
export QUANTAALPHA_VENV_ACTIVATE="${QUANTAALPHA_VENV_ACTIVATE:-/home/workspace/users/liwei/.venv/bin/activate}"

# =============================================================================
# Parse arguments early (run scope may depend on suffix)
# =============================================================================
DIRECTION="$1"
LIBRARY_SUFFIX="$2"
DATA_DOMAINS="$3"
DATA_MODE_ARG="$4"

if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
fi

# =============================================================================
# Activate runtime environment only when explicitly configured
# =============================================================================
activate_runtime_env() {
    if [ -n "${QUANTAALPHA_VENV_ACTIVATE:-}" ]; then
        if [ ! -f "${QUANTAALPHA_VENV_ACTIVATE}" ]; then
            echo "Error: QUANTAALPHA_VENV_ACTIVATE does not exist: ${QUANTAALPHA_VENV_ACTIVATE}"
            exit 1
        fi
        # shellcheck disable=SC1090
        source "${QUANTAALPHA_VENV_ACTIVATE}"
        return 0
    fi

    if [ -n "${CONDA_ENV_NAME:-}" ]; then
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook)" >/dev/null 2>&1
            conda activate "${CONDA_ENV_NAME}" >/dev/null 2>&1 || source activate "${CONDA_ENV_NAME}" >/dev/null 2>&1
        else
            echo "Warning: CONDA_ENV_NAME is set to '${CONDA_ENV_NAME}', but conda is not available in PATH."
        fi
    fi
}

QUANTAALPHA_CMD=(quantaalpha)

if ! command -v quantaalpha >/dev/null 2>&1; then
    activate_runtime_env
fi

if ! command -v quantaalpha >/dev/null 2>&1; then
    if python -c "import quantaalpha.cli" >/dev/null 2>&1; then
        QUANTAALPHA_CMD=(python -m quantaalpha.cli)
    else
        echo "Error: QuantaAlpha CLI is not available."
        echo "Use one of the following approaches:"
        echo "  1) Activate your environment before running this script"
        echo "  2) Set QUANTAALPHA_VENV_ACTIVATE=/path/to/venv/bin/activate"
        echo "  3) Set CONDA_ENV_NAME=<your_env_name>"
        echo "  4) Ensure the active Python can import quantaalpha.cli"
        exit 1
    fi
fi

echo "Python: $(python --version)"
if [ "${QUANTAALPHA_CMD[0]}" = "quantaalpha" ]; then
    echo "QuantaAlpha: $(which quantaalpha)"
else
    echo "QuantaAlpha: ${QUANTAALPHA_CMD[*]}"
fi
echo ""

# =============================================================================
# Experiment isolation
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"${QUANTAALPHA_EXPERIMENT_CONFIG:-configs/experiment.yaml}"}

if [ -z "${EXPERIMENT_ID}" ]; then
    EXP_TS="$(date +%Y%m%d_%H%M%S)"
    EXP_NS="$(date +%N 2>/dev/null || true)"
    case "${EXP_NS}" in
        ""|*N*)
            EXP_NS="${RANDOM}${RANDOM}"
            ;;
    esac
    EXPERIMENT_ID="exp_${EXP_TS}_${EXP_NS}_$$"
fi
export EXPERIMENT_ID

RESULTS_BASE="${DATA_RESULTS_DIR:-./data/results}"
RUNTIME_BASE="${QUANTAALPHA_RUNTIME_DIR:-./data/runtime}"
export DATA_RESULTS_DIR="${RESULTS_BASE}"
export QUANTAALPHA_RUNTIME_DIR="${RUNTIME_BASE}"

# Keep factor libraries compact and store standalone-backtest cache in results.
export FACTOR_LIBRARY_COMPACT="${FACTOR_LIBRARY_COMPACT:-true}"
export FACTOR_LIBRARY_INCLUDE_CODE="${FACTOR_LIBRARY_INCLUDE_CODE:-false}"
export FACTOR_LIBRARY_INCLUDE_TQ_UPSTREAM="${FACTOR_LIBRARY_INCLUDE_TQ_UPSTREAM:-false}"
export FACTOR_LIBRARY_KEEP_RESULT_H5="${FACTOR_LIBRARY_KEEP_RESULT_H5:-false}"
export QUANTAALPHA_SAVE_WORKFLOW_SESSION="${QUANTAALPHA_SAVE_WORKFLOW_SESSION:-false}"
export QUANTAALPHA_CLEAN_RUNTIME_AFTER_RUN="${QUANTAALPHA_CLEAN_RUNTIME_AFTER_RUN:-true}"

# Optional run scope key:
# - set by FACTOR_LIBRARY_SUFFIX to group environment folders
# - defaults to EXPERIMENT_ID for per-run isolation
SCOPE_KEY_RAW="${FACTOR_LIBRARY_SUFFIX:-}"
if [ -n "${SCOPE_KEY_RAW}" ]; then
    SCOPE_KEY_SAFE="$(printf '%s' "${SCOPE_KEY_RAW}" | tr -c 'A-Za-z0-9._-' '_')"
    SCOPE_KEY_SAFE="${SCOPE_KEY_SAFE#_}"
    SCOPE_KEY_SAFE="${SCOPE_KEY_SAFE%_}"
    if [ -z "${SCOPE_KEY_SAFE}" ]; then
        SCOPE_KEY_SAFE="${EXPERIMENT_ID}"
    fi
else
    SCOPE_KEY_SAFE="${EXPERIMENT_ID}"
fi
export QUANTAALPHA_RUN_SCOPE_KEY="${SCOPE_KEY_SAFE}"

if [ "${EXPERIMENT_ID}" != "shared" ]; then
    if [ "${QUANTAALPHA_RUN_SCOPE_KEY}" = "${EXPERIMENT_ID}" ]; then
        export RUN_SCOPE_DIR="${RUNTIME_BASE}/${EXPERIMENT_ID}"
        export EXPERIMENT_RUNTIME_DIR="${RUN_SCOPE_DIR}"
    else
        export RUN_SCOPE_DIR="${RUNTIME_BASE}/${QUANTAALPHA_RUN_SCOPE_KEY}"
        export EXPERIMENT_RUNTIME_DIR="${RUN_SCOPE_DIR}/${EXPERIMENT_ID}"
    fi
    export EXPERIMENT_RESULTS_DIR="${EXPERIMENT_RUNTIME_DIR}"
    export WORKSPACE_PATH="${EXPERIMENT_RUNTIME_DIR}/workspace"
    export PICKLE_CACHE_FOLDER_PATH_STR="${EXPERIMENT_RUNTIME_DIR}/pickle_cache"
    export PROMPT_CACHE_PATH="${EXPERIMENT_RUNTIME_DIR}/prompt_cache.db"
    export LOG_DIR="${EXPERIMENT_RUNTIME_DIR}/log"
    export QUANTAALPHA_LLM_CONV_DIR="${EXPERIMENT_RUNTIME_DIR}/llm_conv"
    export TQ_CANDIDATE_DIR="${EXPERIMENT_RUNTIME_DIR}/tq_candidates"
    export TQ_FACTOR_BASE_DIR="${EXPERIMENT_RUNTIME_DIR}/factor_base"
    export TMPDIR="${EXPERIMENT_RUNTIME_DIR}/tmp"
    # Keep per-run factor-value cache isolated to avoid cross-run cache reuse.
    export FACTOR_CACHE_DIR="${EXPERIMENT_RUNTIME_DIR}/factor_cache"
    export TMP="${TMPDIR}"
    export TEMP="${TMPDIR}"
    mkdir -p "${WORKSPACE_PATH}" \
        "${PICKLE_CACHE_FOLDER_PATH_STR}" \
        "${FACTOR_CACHE_DIR}" \
        "${LOG_DIR}" \
        "${QUANTAALPHA_LLM_CONV_DIR}" \
        "${TQ_CANDIDATE_DIR}" \
        "${TQ_FACTOR_BASE_DIR}" \
        "${TMPDIR}"

    # Reset knowledge base env to avoid accidental cross-run reuse.
    unset FACTOR_CoSTEER_knowledge_base_path FACTOR_CoSTEER_new_knowledge_base_path
    unset FACTOR_COSTEER_KNOWLEDGE_BASE_PATH FACTOR_COSTEER_NEW_KNOWLEDGE_BASE_PATH
    unset CoSTEER_knowledge_base_path CoSTEER_new_knowledge_base_path
    unset COSTEER_KNOWLEDGE_BASE_PATH COSTEER_NEW_KNOWLEDGE_BASE_PATH

    # Optional: explicit warm-start from a chosen knowledge file.
    if [ -n "${QUANTAALPHA_LOAD_KNOWLEDGE_PATH:-}" ]; then
        if [ -f "${QUANTAALPHA_LOAD_KNOWLEDGE_PATH}" ]; then
            export FACTOR_CoSTEER_knowledge_base_path="${QUANTAALPHA_LOAD_KNOWLEDGE_PATH}"
            export FACTOR_COSTEER_KNOWLEDGE_BASE_PATH="${QUANTAALPHA_LOAD_KNOWLEDGE_PATH}"
        else
            echo "Warning: QUANTAALPHA_LOAD_KNOWLEDGE_PATH not found: ${QUANTAALPHA_LOAD_KNOWLEDGE_PATH}"
            echo "         Start from fresh in-memory knowledge for this run."
        fi
    fi

    # Optional: persist current run knowledge to this experiment folder.
    if [ "${QUANTAALPHA_PERSIST_KNOWLEDGE:-false}" = "true" ]; then
        KNOWLEDGE_OUT_PATH="${EXPERIMENT_RUNTIME_DIR}/knowledge_base.pkl"
        export FACTOR_CoSTEER_new_knowledge_base_path="${KNOWLEDGE_OUT_PATH}"
        export FACTOR_COSTEER_NEW_KNOWLEDGE_BASE_PATH="${KNOWLEDGE_OUT_PATH}"
    fi

    echo "Experiment ID: ${EXPERIMENT_ID}"
    echo "Run scope key: ${QUANTAALPHA_RUN_SCOPE_KEY}"
    echo "Run scope dir: ${RUN_SCOPE_DIR}"
    echo "Runtime root: ${EXPERIMENT_RUNTIME_DIR}"
    echo "Results root: ${RESULTS_BASE}"
    echo "Workspace: ${WORKSPACE_PATH}"
    echo "Factor cache dir: ${FACTOR_CACHE_DIR}"
    echo "Prompt cache path: ${PROMPT_CACHE_PATH}"
    echo "Log dir: ${LOG_DIR}"
    echo "Candidate dir: ${TQ_CANDIDATE_DIR}"

    if [ -n "${FACTOR_CoSTEER_knowledge_base_path:-}" ]; then
        echo "Knowledge warm-start: ${FACTOR_CoSTEER_knowledge_base_path}"
    else
        echo "Knowledge warm-start: disabled (fresh run)"
    fi
    if [ -n "${FACTOR_CoSTEER_new_knowledge_base_path:-}" ]; then
        echo "Knowledge persist path: ${FACTOR_CoSTEER_new_knowledge_base_path}"
    else
        echo "Knowledge persist path: disabled"
    fi
fi

# =============================================================================
# TQ upstream defaults
# =============================================================================
export QUANTAALPHA_BACKTEST_ENGINE="${QUANTAALPHA_BACKTEST_ENGINE:-tq}"
if [ -n "${QUANTAALPHA_DATA_ROOT:-}" ] && [ -z "${QUANTAALPHA_SOURCE_DATA_DIR:-}" ]; then
    export QUANTAALPHA_SOURCE_DATA_DIR="${QUANTAALPHA_DATA_ROOT}"
fi

if [ -n "${DATA_DOMAINS}" ]; then
    NORMALIZED_FACTOR_DATA_DOMAINS="$(normalize_factor_data_domains "${DATA_DOMAINS}")"
    if [ $? -ne 0 ]; then
        exit 1
    fi
    if [ -n "${NORMALIZED_FACTOR_DATA_DOMAINS}" ]; then
        export FACTOR_DATA_DOMAINS="${NORMALIZED_FACTOR_DATA_DOMAINS}"
    fi
fi

if [ -n "${DATA_MODE_ARG}" ]; then
    NORMALIZED_FACTOR_DATA_MODE="$(normalize_factor_data_mode "${DATA_MODE_ARG}")"
    if [ $? -ne 0 ] || [ -z "${NORMALIZED_FACTOR_DATA_MODE}" ]; then
        echo "Error: invalid data mode '${DATA_MODE_ARG}'. Expected one of: daily, minutes, fundamental"
        exit 1
    fi
    export FACTOR_DATA_MODE="${NORMALIZED_FACTOR_DATA_MODE}"
elif [ -n "${DATA_DOMAINS}" ]; then
    if [ -z "${FACTOR_DATA_MODE:-}" ]; then
        INFERRED_FACTOR_DATA_MODE="$(infer_factor_data_mode "${DATA_DOMAINS}")"
        if [ -n "${INFERRED_FACTOR_DATA_MODE}" ]; then
            export FACTOR_DATA_MODE="${INFERRED_FACTOR_DATA_MODE}"
        fi
    fi
fi

export FACTOR_BACKTEST_MODE="${FACTOR_BACKTEST_MODE:-single}"

if [ -z "${FACTOR_LIBRARY_SUFFIX:-}" ] && [ "${EXPERIMENT_ID}" != "shared" ]; then
    export FACTOR_LIBRARY_SUFFIX="${EXPERIMENT_ID}"
fi

if [ -n "${FACTOR_LIBRARY_SUFFIX:-}" ]; then
    LIBRARY_FILENAME="all_factors_library_${FACTOR_LIBRARY_SUFFIX}.json"
else
    LIBRARY_FILENAME="all_factors_library.json"
fi
FACTOR_LIBRARY_BASE="${FACTOR_LIBRARY_DIR:-${SCRIPT_DIR}/data/factorlib}"
LIBRARY_PATH="${FACTOR_LIBRARY_BASE}/${LIBRARY_FILENAME}"

echo ""
echo "Starting experiment..."
echo "Config: ${CONFIG_PATH}"
echo "Backtest engine: ${QUANTAALPHA_BACKTEST_ENGINE}"
echo "Data root: ${QUANTAALPHA_DATA_ROOT:-<unset>}"
echo "Results: ${RESULTS_BASE}"
echo "Backtest mode: ${FACTOR_BACKTEST_MODE:-single}"
echo "Factor data mode: ${FACTOR_DATA_MODE:-<config>}"
echo "Factor domains: ${FACTOR_DATA_DOMAINS:-pv}"
echo "----------------------------------------"

cleanup_runtime_artifacts() {
    if [ "${QUANTAALPHA_CLEAN_RUNTIME_AFTER_RUN:-true}" != "true" ]; then
        return 0
    fi

    if [ -z "${EXPERIMENT_RUNTIME_DIR:-}" ] || [ ! -d "${EXPERIMENT_RUNTIME_DIR}" ]; then
        return 0
    fi

    echo ""
    echo "Cleaning runtime artifacts under ${EXPERIMENT_RUNTIME_DIR}"

    local target
    for target in \
        "${EXPERIMENT_RUNTIME_DIR}/workspace" \
        "${EXPERIMENT_RUNTIME_DIR}/pickle_cache" \
        "${EXPERIMENT_RUNTIME_DIR}/llm_conv" \
        "${EXPERIMENT_RUNTIME_DIR}/tq_candidates" \
        "${EXPERIMENT_RUNTIME_DIR}/factor_base" \
        "${EXPERIMENT_RUNTIME_DIR}/tmp" \
        "${EXPERIMENT_RUNTIME_DIR}/knowledge_base.pkl" \
        "${EXPERIMENT_RUNTIME_DIR}/prompt_cache.db"
    do
        if [ -e "${target}" ]; then
            rm -rf "${target}"
        fi
    done
}

if [ -n "${STEP_N}" ]; then
    "${QUANTAALPHA_CMD[@]}" mine --direction "${DIRECTION}" --step_n "${STEP_N}" --config_path "${CONFIG_PATH}"
else
    "${QUANTAALPHA_CMD[@]}" mine --direction "${DIRECTION}" --config_path "${CONFIG_PATH}"
fi

STATUS=$?
cleanup_runtime_artifacts
if [ ${STATUS} -ne 0 ]; then
    exit ${STATUS}
fi

echo ""
echo "Experiment finished."
echo "Factor library: ${LIBRARY_PATH}"
echo "Single-factor standalone backtest from library:"
echo "  python -m quantaalpha.backtest.run_backtest --library-path \"${LIBRARY_PATH}\" --library-factor <factor_id_or_name> --plot true"
echo "Manual single-factor analysis from saved artifacts:"
echo "  python -m quantaalpha.backtest.manual_analysis --factor-value-file data/tq_upstream/candidates/<factor_name>_factor_value.pkl --factor-result-file data/tq_upstream/candidates/<factor_name>_factor_result.pkl --plot true"

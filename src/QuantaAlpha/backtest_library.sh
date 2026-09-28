#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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

if [ $# -lt 1 ]; then
    echo 'Usage: bash backtest_library.sh "<library_name_or_path[,library2,...]>" ["<result_suffix>"]'
    echo 'Examples:'
    echo '  bash backtest_library.sh "pv_1"'
    echo '  bash backtest_library.sh "pv_345_lw,min_34_lw,min_4_pv_67" "rerun_multi"'
    echo '  bash backtest_library.sh "min_12" "recheck_20260416"'
    echo ''
    echo 'Output layout:'
    echo '  single library -> data/tq_upstream/<result_suffix>/<factor_name>/'
    echo '  multi library  -> data/tq_upstream/<result_suffix>/<library_name>/<factor_name>/'
    echo 'Eligibility: a factor is backtested when ANY of train_check_flags.check_passed,'
    echo '             test_check_flags.check_passed, or top-level check_passed/train_passed/test_passed is true.'
    exit 1
fi

if [ -f "${SCRIPT_DIR}/.env" ]; then
    load_env_file "${SCRIPT_DIR}/.env"
else
    echo "Error: .env file not found"
    exit 1
fi

export QUANTAALPHA_VENV_ACTIVATE="${QUANTAALPHA_VENV_ACTIVATE:-/home/workspace/users/liwei/.venv/bin/activate}"

activate_runtime_env

if ! command -v python >/dev/null 2>&1; then
    echo "Error: python is not available after environment activation."
    exit 1
fi

LIBRARY_REF="$1"
OUTPUT_SUFFIX=""
if [ $# -ge 2 ]; then
    OUTPUT_SUFFIX="$2"
fi

echo "Python: $(python --version)"
echo "Library or libraries: ${LIBRARY_REF}"
echo "Output suffix: ${OUTPUT_SUFFIX:-<default>}"
echo "Eligibility: train OR test check_passed = true (default Python behaviour)"
echo "Artifacts: diagnostics plot + quality report + notebook report"
echo ""

cmd=(
    python -m quantaalpha.backtest.run_library_backtest
    --library "${LIBRARY_REF}"
)
if [ -n "${OUTPUT_SUFFIX}" ]; then
    cmd+=(--output_suffix "${OUTPUT_SUFFIX}")
fi
"${cmd[@]}"

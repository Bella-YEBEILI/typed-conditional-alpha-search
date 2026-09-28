#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ $# -lt 2 ]; then
    cat <<'USAGE'
Usage: bash collect_factor_code.sh <domain> "<suffix1[,suffix2,...]>" [output_root]

  <domain>          target subdirectory under output/, must be one of: pv | minutes | joint
  <suffix1,...>     one or more directory names under data/tq_upstream/ (comma-separated)
  [output_root]     optional override of the output base directory (default: ./output)

Examples:
  bash collect_factor_code.sh pv      "pv_678"
  bash collect_factor_code.sh minutes "min_34_lw,min_12_20260421_minute_fixed_v2"
  bash collect_factor_code.sh joint   "min_4_pv_67_fix"

Behaviour:
  - walks each suffix dir under data/tq_upstream/<suffix> recursively
  - copies every <factor_name>.py to output/<domain>/<factor_name>.py
  - on filename collision, appends the suffix dir name to keep both copies
  - non-.py artefacts (json / pkl / png / ipynb) are ignored
USAGE
    exit 1
fi

DOMAIN="$1"
SUFFIX_LIST="$2"
OUTPUT_ROOT="${3:-${SCRIPT_DIR}/output}"

case "${DOMAIN}" in
    pv|minutes|joint) ;;
    *)
        echo "Error: <domain> must be one of: pv | minutes | joint (got '${DOMAIN}')"
        exit 1
        ;;
esac

UPSTREAM_ROOT="${SCRIPT_DIR}/data/tq_upstream"
if [ ! -d "${UPSTREAM_ROOT}" ]; then
    echo "Error: upstream directory not found: ${UPSTREAM_ROOT}"
    exit 1
fi

DEST_DIR="${OUTPUT_ROOT}/${DOMAIN}"
mkdir -p "${DEST_DIR}"

echo "Domain      : ${DOMAIN}"
echo "Suffix list : ${SUFFIX_LIST}"
echo "Source root : ${UPSTREAM_ROOT}"
echo "Destination : ${DEST_DIR}"
echo ""

total_copied=0
total_skipped=0
total_renamed=0

IFS=',' read -r -a SUFFIX_ARR <<< "${SUFFIX_LIST}"
for raw_suffix in "${SUFFIX_ARR[@]}"; do
    suffix="${raw_suffix#"${raw_suffix%%[![:space:]]*}"}"
    suffix="${suffix%"${suffix##*[![:space:]]}"}"
    if [ -z "${suffix}" ]; then
        continue
    fi

    src_dir="${UPSTREAM_ROOT}/${suffix}"
    if [ ! -d "${src_dir}" ]; then
        echo "  [skip] suffix '${suffix}': directory not found at ${src_dir}"
        total_skipped=$((total_skipped + 1))
        continue
    fi

    echo "  [scan] ${src_dir}"
    copied_in_suffix=0
    while IFS= read -r -d '' py_file; do
        base_name="$(basename "${py_file}")"
        factor_name="${base_name%.py}"
        parent_dir_name="$(basename "$(dirname "${py_file}")")"
        if [ "${parent_dir_name}" != "${factor_name}" ]; then
            continue
        fi

        dest_path="${DEST_DIR}/${base_name}"
        if [ -e "${dest_path}" ]; then
            existing_hash="$(sha1sum "${dest_path}" | awk '{print $1}')"
            new_hash="$(sha1sum "${py_file}" | awk '{print $1}')"
            if [ "${existing_hash}" = "${new_hash}" ]; then
                continue
            fi
            renamed_path="${DEST_DIR}/${factor_name}__${suffix}.py"
            cp -f "${py_file}" "${renamed_path}"
            echo "    [rename] ${factor_name}.py -> ${factor_name}__${suffix}.py (hash mismatch with existing)"
            total_renamed=$((total_renamed + 1))
        else
            cp -f "${py_file}" "${dest_path}"
        fi
        copied_in_suffix=$((copied_in_suffix + 1))
    done < <(find "${src_dir}" -type f -name '*.py' -print0)

    echo "    copied ${copied_in_suffix} factor file(s) from suffix '${suffix}'"
    total_copied=$((total_copied + copied_in_suffix))
done

echo ""
echo "Summary:"
echo "  copied   : ${total_copied}"
echo "  renamed  : ${total_renamed}  (collision with different content)"
echo "  skipped  : ${total_skipped}  (suffix dir not found)"
echo "  output   : ${DEST_DIR}"

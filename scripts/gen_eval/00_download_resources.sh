#!/usr/bin/env bash
# Download the resources this project actually uses. Nothing here auto-runs;
# call it explicitly. Everything lands under $ATS_DATA_ROOT (outside git).
#
# Usage:
#   scripts/gen_eval/00_download_resources.sh [p12|bits|timeimm|all]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"

WHAT="${1:-p12}"
DATA="${ATS_DATA_ROOT}"
mkdir -p "${DATA}"

dl() {  # url outfile  (resume-capable retry loop)
    local url="$1" out="$2"
    for i in $(seq 1 30); do
        curl -s -C - -fL -o "${out}" "${url}" && return 0
        echo "retry $i for ${url} ($(stat -c%s "${out}" 2>/dev/null || echo 0) bytes)" | tee -a "${LOG_FILE}"
    done
    echo "FAILED: ${url}" | tee -a "${LOG_FILE}"; return 1
}

download_p12() {
    local d="${DATA}/data/p12"; mkdir -p "${d}"
    echo "== PhysioNet Challenge 2012 Set A" | tee -a "${LOG_FILE}"
    dl "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz"   "${d}/set-a.tar.gz"
    dl "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt" "${d}/outcomes-a.txt"
    tar xzf "${d}/set-a.tar.gz" -C "${d}"
    echo "records: $(ls "${d}/set-a" | wc -l)" | tee -a "${LOG_FILE}"
    sha256sum "${d}/set-a.tar.gz" | tee -a "${LOG_FILE}"
}

download_bits() {
    # source snapshot only; the full BITS-data download is deliberately skipped
    # (upstream pins py3.10 + old torch). See src/agent4ts/data/bits.py.
    local d="${DATA}/vendor"; mkdir -p "${d}"
    echo "== BITS source snapshot (source only, no dataset)" | tee -a "${LOG_FILE}"
    dl "https://anonymous.4open.science/api/repo/BITS-8F2E/zip" "${d}/BITS-8F2E.zip"
    unzip -q -o "${d}/BITS-8F2E.zip" -d "${d}/bits_tmp"
    echo "BITS files: $(find "${d}/bits_tmp" -type f | wc -l)" | tee -a "${LOG_FILE}"
    sha256sum "${d}/BITS-8F2E.zip" | tee -a "${LOG_FILE}"
}

download_timeimm() {
    # Time-IMM EPA-Air processed CSVs, fetched from the GitHub mirror.
    local d="${DATA}/data/timeimm"; mkdir -p "${d}"
    echo "== Time-IMM EPA-Air" | tee -a "${LOG_FILE}"
    for city in Los_Angeles Dallas Denver; do
        dl "${GIT_MIRROR}https://github.com/blacksnail789521/Time-IMM/raw/master/data/EPA-Air/processed/${city}.csv" \
           "${d}/${city}.csv" || true
    done
    ls -la "${d}" | tee -a "${LOG_FILE}"
}

case "${WHAT}" in
    p12)     download_p12 ;;
    bits)    download_bits ;;
    timeimm) download_timeimm ;;
    all)     download_p12; download_bits; download_timeimm ;;
    *) echo "unknown target ${WHAT}"; exit 2 ;;
esac
echo "== done ($(date '+%F %T'))" | tee -a "${LOG_FILE}"

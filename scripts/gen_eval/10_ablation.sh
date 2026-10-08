#!/usr/bin/env bash
# no_vision / no_semantics / one_shot ablations (each is a separate run dir).
set -euo pipefail
export ATS_KIND="${ATS_KIND:-gen_eval}"; export ATS_LOG_NAME=ablation
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
for ab in no_vision no_semantics one_shot; do
    echo "== ablation ${ab}" | tee -a "${LOG_FILE}"
    ats_run_cli --config "${ATS_CONFIG}" --kind "${ATS_KIND}" search --ablation "${ab}" "$@"
done

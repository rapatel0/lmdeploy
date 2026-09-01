#!/usr/bin/env bash
# Test SGLang's input-token anchor contract under graph capture and audited identity.
set -euo pipefail
export TM_DFLASH_SGLANG_INPUT_ANCHOR=1
exec bash /job/dflash_tilelang_graph_identity_job.sh

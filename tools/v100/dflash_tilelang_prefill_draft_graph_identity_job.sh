#!/usr/bin/env bash
# Draft immediately after the completed prompt prefill, matching SGLang lifecycle.
set -euo pipefail
export TM_DFLASH_DRAFT_AFTER_PREFILL=1
exec bash /job/dflash_tilelang_graph_identity_job.sh

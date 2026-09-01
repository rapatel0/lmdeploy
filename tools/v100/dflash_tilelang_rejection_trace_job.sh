#!/usr/bin/env bash
# Explain the first production verifier rejections without changing decisions.
set -euo pipefail
export TM_SPEC_REJECT_DUMP=1
exec bash /job/dflash_tilelang_graph_identity_job.sh

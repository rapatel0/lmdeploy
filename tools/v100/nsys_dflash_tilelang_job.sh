#!/usr/bin/env bash
# Profile the exact opt-in TileLang DFlash verifier route.
set -euo pipefail
export TM_DFLASH_TILELANG_DRAFT_ATTENTION=1
export TM_DFLASH_DRAFT_GRAPH=1
export TM_DFLASH_PERSISTENT_WORKSPACE=1
export TM_DFLASH_LOCAL_TOPK=0
export TM_DFLASH_PAGED_Q8=0
export TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1
exec bash /src/tools/v100/nsys_dflash_job.sh

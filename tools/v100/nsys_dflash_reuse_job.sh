#!/usr/bin/env bash
# Profile the installed current-default wheel without rebuilding it.
set -euo pipefail
export REUSE_WHEEL=1
exec bash /src/tools/v100/nsys_dflash_job.sh

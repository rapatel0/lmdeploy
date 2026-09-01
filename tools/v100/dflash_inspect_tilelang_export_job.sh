#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
for path in Path('/results').rglob('*_host.cc'):
    if path.name not in {'partial_host.cc', 'combine_host.cc'}:
        continue
    print(f'=== {path} ===')
    text = path.read_text(errors='replace')
    for needle in ('cuLaunchKernel', 'void* args', 'void *args', 'kernel_args'):
        start = 0
        while True:
            at = text.find(needle, start)
            if at < 0:
                break
            print(text[max(0, at - 2000):at + 2500])
            start = at + len(needle)
PY

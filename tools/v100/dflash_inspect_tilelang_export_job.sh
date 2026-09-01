#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
for path in Path('/results').rglob('*_host.cc'):
    if path.name not in {'partial_host.cc', 'combine_host.cc'}:
        continue
    print(f'=== {path} ===')
    text = path.read_text(errors='replace')
    start = text.find('void* Q =') if path.name == 'partial_host.cc' else text.find('void* Partial_O =')
    end = text.find('TVMFFIFunctionCall(main_kernel_packed', start)
    if start >= 0 and end >= 0:
        print(text[start : end + 300])
    else:
        print('FAIL: launch argument section not found')
PY

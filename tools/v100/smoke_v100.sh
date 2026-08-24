#!/usr/bin/env bash
# Smoke-test a built LMDeploy wheel on a real V100.
#
# Compiling for SM70 does not prove the wheel runs. This checks the three
# things that actually gate the campaign:
#
#   1. torch itself ships SM70 kernels. A torch built for CC 7.5 and above
#      loads fine and then fails at the first kernel launch, so check
#      torch.cuda.get_arch_list() rather than trusting the install.
#   2. _turbomind imports. It is a top-level module found through a sys.path
#      insert into lmdeploy/lib, not a submodule of lmdeploy.turbomind.
#   3. A CUDA kernel actually executes on the device.
#
# The script fails on the first error. An earlier version ended with a
# pipeline that swallowed the real status, so the Job reported success while
# the import had failed.
set -euo pipefail

echo "=== device ==="
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader

echo "=== torch SM70 support ==="
python - <<'PY'
import sys
import torch

arch = torch.cuda.get_arch_list()
print('torch', torch.__version__)
print('arch_list', arch)
if 'sm_70' not in arch:
    print('FAIL: this torch has no sm_70 kernels', file=sys.stderr)
    print('it will load and then fail at the first kernel launch', file=sys.stderr)
    sys.exit(1)
cap = torch.cuda.get_device_capability(0)
if cap != (7, 0):
    print(f'FAIL: expected compute capability (7, 0), found {cap}', file=sys.stderr)
    sys.exit(1)
print('PASS: torch ships sm_70 and the device is (7, 0)')
PY

echo "=== execute a kernel on the device ==="
python - <<'PY'
import sys
import torch

# Half precision, because the campaign runs FP16 on SM70.
a = torch.randn(512, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 512, device='cuda', dtype=torch.float16)
c = (a @ b).float()
torch.cuda.synchronize()
if not torch.isfinite(c).all():
    print('FAIL: matmul produced non-finite values', file=sys.stderr)
    sys.exit(1)
print('PASS: fp16 matmul executed, result finite')
PY

echo "=== import turbomind ==="
python - <<'PY'
import os.path as osp
import sys

import lmdeploy

# Mirror what lmdeploy/turbomind/turbomind.py does.
sys.path.append(osp.join(osp.split(lmdeploy.__file__)[0], 'lib'))
import _turbomind as tm  # noqa: E402

names = [n for n in dir(tm) if not n.startswith('_')]
print('PASS: _turbomind imported')
print('exported names:', len(names))
PY

echo "=== turbomind engine entrypoint ==="
python -c "from lmdeploy.turbomind import TurboMind; print('PASS: TurboMind importable')"

echo
echo "ALL SMOKE CHECKS PASSED"

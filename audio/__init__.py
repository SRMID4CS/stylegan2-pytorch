"""Cross-repo audio contract package (shared verbatim with dlg-sonic).

- `meldataset.py` / `env.py`: vendored VERBATIM from NVIDIA/BigVGAN (pinned commit in
  `contract.BIGVGAN_COMMIT`). `meldataset.py` uses a top-level `from env import AttrDict`,
  so this package puts its own directory on sys.path before the import resolves.
- `contract.py`: the mel config, global affine, and canvas embed/crop — the single
  source of truth for the checkpoint sidecar values.
- `bigvgan/`: vendored BigVGAN inference code (vocoder only; not part of the contract
  copy that goes to dlg-sonic).
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

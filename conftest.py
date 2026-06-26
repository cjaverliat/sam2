# SPDX-License-Identifier: Apache-2.0
# Ensure the repo root is on sys.path so `import sam` resolves the src
# directory regardless of whether the editable install has been updated
# (pyproject.toml packaging is updated in Task 4 of the sam2->sam refactor).
import sys
from pathlib import Path

_root = str(Path(__file__).parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

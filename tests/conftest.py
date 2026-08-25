"""Make the src packages importable for tests without installation.

All test imports use the full ``src.xxx`` prefix (src/ has no ``__init__.py``,
so ``import src.data.manifest`` needs the repo root on ``sys.path``).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

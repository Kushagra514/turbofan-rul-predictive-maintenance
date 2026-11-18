"""
Pytest configuration and environment setup for Turbofan RUL tests.
"""

import sys
from pathlib import Path

# Add project root to sys.path so src and scripts can be imported cleanly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sys
import os
from pathlib import Path

# Add root directory to sys.path for test resolution
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

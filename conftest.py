import sys
from pathlib import Path

# Insert project root directory into sys.path for pytest
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

import sys
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DATA_DIR

def load_csv(filename: str, directory: Path = DATA_DIR) -> pd.DataFrame:
    """Load a CSV file with all string types."""
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, dtype=str)

import os
from pathlib import Path

from pint import UnitRegistry

REPO_DIR = Path(__file__).parents[1]
os.environ.setdefault("DATA_DIR", str(REPO_DIR / "data"))
DATA_DIR = Path(os.environ["DATA_DIR"])
FRAGMENT_DATA_DIR = DATA_DIR / "fragments"
SCROLL_DATA_DIR = DATA_DIR / "scrolls"

ureg = UnitRegistry()
Q_ = ureg.Quantity

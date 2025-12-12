"""Some type definitions"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StrArray = NDArray[np.str_]

PathType = str | Path

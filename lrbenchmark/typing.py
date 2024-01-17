from pathlib import Path
from typing import Tuple, Union

import numpy as np


XYType = Tuple[np.ndarray, np.ndarray]
TrainTestPair = Tuple[XYType, XYType]
PathLike = Union[str, Path]

from collections import namedtuple
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
Result = namedtuple('Result', 'metrics figures holdout_lrs')

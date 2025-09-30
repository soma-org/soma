# arrgen.pyi
from typing import List
import numpy as np

def uniform_array(
    seed: int, shape: List[int], min: float, max: float
) -> np.ndarray: ...
def normal_array(
    seed: int, shape: List[int], mean: float, std_dev: float
) -> np.ndarray: ...
def constant_array(shape: List[int], value: float) -> np.ndarray: ...

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class Module(ABC):
    @abstractmethod
    def __call__(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
from typing import (
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as _np

IndexT = Union[int, slice, list, _np.ndarray]
"""Type alias for an index in a list or numpy array."""

_BaseSingleItemT = Tuple[_np.ndarray, int, str, _np.ndarray]
_BaseMultipleItemT = List[Tuple[_np.ndarray, int, str, _np.ndarray]]
BaseItemT = Union[
    _BaseSingleItemT,
    _BaseMultipleItemT
]
"""Type alias for a single or multiple BaseImageClassificationDataset items. It is a tuple containing the 
image data, label index, category name, and the RGB histogram.
"""

BaseIterT = Generator[_BaseSingleItemT, None, None]
"""Type alias for an iterator of BaseImageClassificationDataset items. 
"""

_LBPSingleItemT = Tuple[_np.ndarray, int, str, _np.ndarray, _np.ndarray, _np.ndarray, Optional[_np.ndarray]]
_LBPMultipleItemT = List[Tuple[_np.ndarray, int, str, _np.ndarray, _np.ndarray, _np.ndarray, Optional[_np.ndarray]]]
LBPItemT = Union[
    _LBPSingleItemT,
    _LBPMultipleItemT
]
"""Type alias for a single or multiple LBPImageClassificationDataset items. It is a tuple containing the
image data, label index, category name, the RGB histogram, the LBP graph, the LBP histogram and the feature vector
(if mixed).
"""
LBPIterT = Generator[_LBPSingleItemT, None, None]
"""Type alias for an iterator of LBPImageClassificationDataset items."""

DEFAULT_FONT = 'Times New Roman'
RANDOM_SEED = 42
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

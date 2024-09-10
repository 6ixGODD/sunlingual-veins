from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Sized,
    Tuple,
    Union,
)

import cv2 as _cv2
import matplotlib.gridspec as _gd
import matplotlib.pyplot as _plt

from ._types import (
    BaseItemT as _ItemT,
    BaseIterT as _IterT,
    DEFAULT_FONT as _DEFAULT_FONT,
    IMAGE_EXTENSIONS as _IMAGE_EXTENSIONS,
    IndexT as _IndexT,
    RANDOM_SEED as _RANDOM_SEED,
)

_plt.rcParams['font.family'] = _DEFAULT_FONT
import numpy as _np

_np.random.seed(_RANDOM_SEED)

from .. import _utils


class BaseImageClassificationDataset(Iterable, Sized):
    _images: List[_np.ndarray] = []
    _labels: List[int] = []
    _categories: Dict[int, str] = {}
    _rgb_hist: List[_np.ndarray] = []

    @classmethod
    def from_directory(
        cls,
        root: Union[str, Path],
        *,
        limit: int = 0,
        categories: List[str] = None
    ) -> 'BaseImageClassificationDataset':
        dataset = cls()
        if categories is not None:
            dataset._categories = {i: c for i, c in enumerate(categories)}
        dataset._load_images(root, limit=limit)
        return dataset

    def __len__(self) -> int:
        return self._images.__len__()

    def __getitem__(self, idx_: _IndexT) -> _ItemT:
        if isinstance(idx_, int):
            return (
                self._images[idx_],
                self._labels[idx_],
                self._categories[self._labels[idx_]],
                self._rgb_hist[idx_]
            )
        elif isinstance(idx_, slice):
            return [
                (
                    self._images[i],
                    self._labels[i],
                    self._categories[self._labels[i]],
                    self._rgb_hist[i]
                ) for i in range(*idx_.indices(len(self)))
            ]
        elif isinstance(idx_, list) or isinstance(idx_, _np.ndarray):
            return [
                (
                    self._images[i],
                    self._labels[i],
                    self._categories[self._labels[i]],
                    self._rgb_hist[i]
                ) for i in idx_
            ]
        else:
            raise ValueError(f"Invalid index type {type(idx_)}")

    def __setitem__(self, idx_: _IndexT, item_: _ItemT) -> None:
        if isinstance(idx_, int):
            (
                self._images[idx_],
                self._labels[idx_],
                self._categories[self._labels[idx_]],
                self._rgb_hist[idx_]
            ) = item_
        elif isinstance(idx_, slice):
            for i, j in zip(range(*idx_.indices(len(self))), item_):
                (
                    self._images[i],
                    self._labels[i],
                    self._categories[self._labels[i]],
                    self._rgb_hist[i]
                ) = j
        elif isinstance(idx_, list) or isinstance(idx_, _np.ndarray):
            for i, j in zip(idx_, item_):
                (
                    self._images[i],
                    self._labels[i],
                    self._categories[self._labels[i]],
                    self._rgb_hist[i]
                ) = j
        else:
            raise ValueError(f"Invalid index type {type(idx_)}")

    def __iter__(self) -> _IterT:
        for i in range(self.__len__()):
            yield self[i]

    def __add__(
        self,
        other_: 'BaseImageClassificationDataset'
    ) -> 'BaseImageClassificationDataset':
        if self._categories != other_._categories:
            raise ValueError(f"Categories mismatch: {self._categories} != {other_._categories}")
        new_dataset = BaseImageClassificationDataset()
        new_dataset._images = self._images + other_._images
        new_dataset._labels = self._labels + other_._labels
        new_dataset._categories = self._categories
        return new_dataset

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'size={len(self)}, '
            f'num_classes={len(self._categories)}, '
            f'categories={self._categories})'
        )

    def __repr__(self) -> str:
        return self.__str__()

    def overview(
        self,
        *,
        save_path: Union[str, Path] = 'overview.png'
    ) -> None:
        figure = _plt.figure(figsize=(10, 6.5), dpi=600)
        gs = _gd.GridSpec(1, 2, width_ratios=[1, 1])

        ax0 = figure.add_subplot(gs[0])
        labels, counts = _np.unique(self._labels, return_counts=True)
        categories = [self._categories[lb] for lb in labels]
        ax0.bar(categories, counts)
        ax0.set_xticks(categories)
        _plt.xticks(rotation=45, fontstyle='italic', fontsize=10)
        ax0.set_xlabel('Class')
        ax0.set_ylabel('Count')
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)
        ax0.set_title('Image Distribution')

        ax1 = figure.add_subplot(gs[1])
        inner_gs = _gd.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1], wspace=0.1, hspace=0.1)
        for i in range(9):
            ax = figure.add_subplot(inner_gs[i])
            img_index = _np.random.randint(len(self._images))
            ax.imshow(self._images[img_index])
            ax.set_title(f'Class: {self._categories[self._labels[img_index]]}', fontsize=10)
            ax.axis('off')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.axis('off')

        _plt.tight_layout()
        _plt.savefig(save_path.__str__())

    def overview_image_distribution(
        self,
        *,
        save_path: Union[str, Path] = 'image_distribution.png'
    ) -> None:
        _plt.figure(figsize=(10, 10), dpi=600)
        labels, counts = _np.unique(self._labels, return_counts=True)
        categories = [self._categories[lb] for lb in labels]
        _plt.bar(categories, counts)
        _plt.xticks(rotation=45, fontstyle='italic', fontsize=10)
        _plt.xlabel('Class')
        _plt.ylabel('Count')
        _plt.title('Image Distribution')
        _plt.savefig(save_path.__str__())

    def overview_images(
        self,
        *,
        save_path: Union[str, Path] = 'overview_images.png'
    ) -> None:
        _plt.figure(figsize=(10, 10), dpi=600)
        gs = _gd.GridSpec(3, 3, wspace=0.1, hspace=0.1)
        for i in range(9):
            ax = _plt.subplot(gs[i])
            img_index = _np.random.randint(len(self._images))
            ax.imshow(self._images[img_index])
            ax.set_title(f'Class: {self._categories[self._labels[img_index]]}', fontsize=10)
            ax.axis('off')
        _plt.tight_layout()
        _plt.savefig(save_path.__str__())

    def shuffle(self) -> None:
        indices = _np.arange(len(self))
        _np.random.shuffle(indices)
        self._images = [self._images[i] for i in indices]
        self._labels = [self._labels[i] for i in indices]

    def resize(
        self,
        size: Tuple[int, int],
        *,
        padding: bool = False,
        **kwargs: Any
    ) -> None:
        for i, image in enumerate(self._images):
            if padding:
                self._images[i], _ = _utils.padding_resize(image, size=size, **kwargs)
            else:
                self._images[i] = _cv2.resize(image, size)

    def append(
        self,
        *,
        image: _np.ndarray,
        label: int,
        category: str = '',
    ) -> None:
        if category != '' and label not in self._categories:
            if category in self._categories.values():
                raise ValueError(f"Category {category} not match with label {label}")
            self._categories[label] = category
        self._images.append(image)
        self._labels.append(label)
        self._rgb_hist.append(self.calculate_rgb_histogram(image))

    def save_images(
        self,
        output_dir: Union[str, Path],
        *,
        fmt: str = 'jpg',
        split: bool = False,
        split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    ) -> None:
        if f'.{fmt}' not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid image format {fmt}")
        output_dir = Path(output_dir)
        if split:
            for i, (image, label, category, *_) in enumerate(self):
                (output_dir / 'train' / category).mkdir(parents=True, exist_ok=True)
                (output_dir / 'val' / category).mkdir(parents=True, exist_ok=True)
                (output_dir / 'test' / category).mkdir(parents=True, exist_ok=True)
                split_idx = i % len(split_ratio)
                if split_idx == 0:
                    split_dir = 'train'
                elif split_idx == 1:
                    split_dir = 'val'
                else:
                    split_dir = 'test'
                _cv2.imwrite(
                    (output_dir / split_dir / category / f'{label}-{i}.{fmt}').absolute().__str__(),
                    image
                )
        else:
            for i, (image, label, category, *_) in enumerate(self):
                if not Path(output_dir / category).exists():
                    Path(output_dir / category).mkdir(parents=True, exist_ok=True)
                _cv2.imwrite(
                    str(output_dir / category / f"{label}-{i}.{fmt}"),
                    image
                )

    def _load_images(
        self,
        root: Union[str, Path],
        *,
        limit: int = 0
    ) -> None:
        categories = [f.name for f in Path(root).iterdir() if f.is_dir()]
        num_classes = categories.__len__()
        for i, c in enumerate(categories):
            path = Path(root) / c
            for j, f in enumerate(path.glob('*')):
                if f.suffix in _IMAGE_EXTENSIONS:
                    self.append(
                        image=_cv2.imread(str(f)),
                        label=i,
                        category=c,
                    )
                    if limit != 0 and j >= limit / num_classes:
                        break

    @staticmethod
    def calculate_rgb_histogram(image: _np.ndarray) -> _np.ndarray:
        """
        Calculate the RGB histogram of an image.

        This function computes the histogram of the red, green, and blue channels of an image separately.
        The histograms are normalized so that the sum of all bin values is 1.

        Args:
            image (np.ndarray): The input image. Must be a 3D numpy array with three channels (RGB).

        Returns:
            np.ndarray: The concatenated RGB histogram, a 1D numpy array of shape (256 * 3,).
        """

        # Check if the input image has three channels
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3D array with three channels (RGB).")

        # Calculate histograms for each channel (R, G, B)
        hist_r, _ = _np.histogram(image[:, :, 0], bins=_np.arange(0, 257), range=(0, 256))
        hist_g, _ = _np.histogram(image[:, :, 1], bins=_np.arange(0, 257), range=(0, 256))
        hist_b, _ = _np.histogram(image[:, :, 2], bins=_np.arange(0, 257), range=(0, 256))

        # Concatenate the histograms
        hist = _np.concatenate([hist_r, hist_g, hist_b]).astype("float")

        # Normalize the histogram
        hist /= (hist.sum() + 1e-06)

        return hist

    def reduce_rgb_dimensionality(self, method: str, **kwargs: Any) -> None:
        if method == 'pca':
            self._rgb_hist = _utils.pca_transform(self._rgb_hist, **kwargs)
        elif method == 'tsne':
            self._rgb_hist = _utils.tsne_transform(self._rgb_hist, **kwargs)
        else:
            raise NotImplementedError(f"Dimensionality reduction method {method} not implemented")

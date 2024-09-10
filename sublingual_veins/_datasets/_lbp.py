import logging
import warnings
from pathlib import Path
from typing import (Any, Dict, List, Tuple, Type, Union)

import cv2 as _cv2
import matplotlib.gridspec as _gd
import matplotlib.pyplot as _plt
from typing_extensions import Iterable

from ._base import BaseImageClassificationDataset
from ._types import (
    DEFAULT_FONT as _DEFAULT_FONT,
    IMAGE_EXTENSIONS as _IMAGE_EXTENSIONS,
    IndexT as _IndexT,
    LBPItemT as _LBPItemT,
    LBPIterT as _LBPIterT,
    RANDOM_SEED as _RANDOM_SEED,
)

_plt.rcParams['font.family'] = _DEFAULT_FONT
import numpy as _np

_np.random.seed(_RANDOM_SEED)

from .. import _utils

_LOGGER = logging.getLogger(__name__)


class LocalBinaryPatternsImageClassificationDataset(BaseImageClassificationDataset):
    _images: List[_np.ndarray] = []
    _labels: List[int] = []
    _categories: Dict[int, str] = {}
    _rgb_hist: List[_np.ndarray] = []
    _with_channel_flatten: bool = False
    _lbp_images: List[_np.ndarray] = []
    _lbp_hist: List[_np.ndarray] = []
    _feature_vec: List[_np.ndarray] = []

    @classmethod
    def from_directory(
        cls: Type['LocalBinaryPatternsImageClassificationDataset'],
        root: Union[str, Path],
        *,
        limit: int = 0,
        categories: List[str] = None,
        channel_flatten: bool = False
    ) -> 'LocalBinaryPatternsImageClassificationDataset':
        dataset = cls()
        if categories is not None:
            dataset._categories = {i: c for i, c in enumerate(categories)}
        dataset._load_images(root, limit=limit, channel_flatten=channel_flatten)
        return dataset

    @classmethod
    def from_base_dataset(
        cls: Type['LocalBinaryPatternsImageClassificationDataset'],
        dataset: BaseImageClassificationDataset,
        channel_flatten: bool = False
    ) -> 'LocalBinaryPatternsImageClassificationDataset':
        new_dataset = cls()
        new_dataset._from_base_dataset(dataset, channel_flatten=channel_flatten)
        return new_dataset

    def __getitem__(self, idx_: _IndexT) -> _LBPItemT:
        if isinstance(idx_, int):
            return (
                self._images[idx_],
                self._labels[idx_],
                self._categories[self._labels[idx_]],
                self._rgb_hist[idx_],
                self._lbp_images[idx_],
                self._lbp_hist[idx_],
                self._feature_vec[idx_]
            )
        elif isinstance(idx_, slice):
            return [
                (
                    self._images[i],
                    self._labels[i],
                    self._categories[self._labels[i]],
                    self._rgb_hist[i],
                    self._lbp_images[i],
                    self._lbp_hist[i],
                    self._feature_vec[i]
                ) for i in range(*idx_.indices(len(self)))
            ]
        elif isinstance(idx_, list) or isinstance(idx_, _np.ndarray):
            return [
                (
                    self._images[i],
                    self._labels[i],
                    self._categories[self._labels[i]],
                    self._rgb_hist[i],
                    self._lbp_images[i],
                    self._lbp_hist[i],
                    self._feature_vec[i]
                ) for i in idx_
            ]
        else:
            raise ValueError(f"Invalid index type {type(idx_)}")

    def __setitem__(self, idx_: _IndexT, value: _LBPItemT) -> None:
        if isinstance(idx_, int):
            self._images[idx_] = value[0]
            self._labels[idx_] = value[1]
            self._categories[value[1]] = value[2]
            self._rgb_hist[idx_] = value[3]
            self._lbp_images[idx_] = value[4]
            self._lbp_hist[idx_] = value[5]
            self._feature_vec[idx_] = value[6]
        elif isinstance(idx_, slice):
            for i, v in zip(range(*idx_.indices(len(self))), value):
                self._images[i] = v[0]
                self._labels[i] = v[1]
                self._categories[v[1]] = v[2]
                self._rgb_hist[i] = v[3]
                self._lbp_images[i] = v[4]
                self._lbp_hist[i] = v[5]
                self._feature_vec[i] = v[6]
        elif isinstance(idx_, list) or isinstance(idx_, _np.ndarray):
            for i, v in zip(idx_, value):
                self._images[i] = v[0]
                self._labels[i] = v[1]
                self._categories[v[1]] = v[2]
                self._rgb_hist[i] = v[3]
                self._lbp_images[i] = v[4]
                self._lbp_hist[i] = v[5]
                self._feature_vec[i] = v[6]
        else:
            raise ValueError(f"Invalid index type {type(idx_)}")

    def __iter__(self) -> _LBPIterT:
        for i in range(len(self)):
            yield self[i]

    def __add__(
        self,
        other_: 'LocalBinaryPatternsImageClassificationDataset'
    ) -> 'LocalBinaryPatternsImageClassificationDataset':
        if self._categories != other_._categories:
            raise ValueError(f"Categories mismatch: {self._categories} != {other_._categories}")
        if not self._with_channel_flatten and other_._with_channel_flatten:
            raise ValueError("Cannot merge dataset with channel flatten and dataset without channel flatten.")
        new_dataset = LocalBinaryPatternsImageClassificationDataset()
        new_dataset._images = self._images + other_._images
        new_dataset._labels = self._labels + other_._labels
        new_dataset._categories = self._categories
        new_dataset._lbp_images = self._lbp_images + other_._lbp_images
        new_dataset._lbp_hist = self._lbp_hist + other_._lbp_hist
        new_dataset._with_channel_flatten = self._with_channel_flatten
        return new_dataset

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'size={len(self)}, '
            f'num_classes={len(self._categories)}, '
            f'categories={self._categories}'
            f'with_channel_flatten={self._with_channel_flatten})'
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def lbp_hist(self) -> List[_np.ndarray]:
        return self._lbp_hist

    @property
    def labels(self) -> List[int]:
        return self._labels

    def _from_base_dataset(
        self,
        dataset: BaseImageClassificationDataset,
        *,
        channel_flatten: bool = False
    ) -> None:
        self._images = dataset._images
        self._labels = dataset._labels
        self._categories = dataset._categories
        for image in self._images:
            if channel_flatten:
                R, G, B = _cv2.split(image)
                gray_img = _np.hstack((R, G, B))
            else:
                gray_img = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
            lbp_image = self.faster_calculate_lbp(gray_img)
            lbp_hist = self.calculate_lbp_hist(lbp_image)
            self._lbp_images.append(lbp_image)
            self._lbp_hist.append(lbp_hist)

    def shuffle(self) -> None:
        indices = _np.arange(len(self))
        _np.random.shuffle(indices)
        self._images = [self._images[i] for i in indices]
        self._labels = [self._labels[i] for i in indices]
        self._lbp_images = [self._lbp_images[i] for i in indices]
        self._lbp_hist = [self._lbp_hist[i] for i in indices]

    def append(self, *, image: _np.ndarray, label: int, category: str = '', **kwargs: Any) -> None:
        super(LocalBinaryPatternsImageClassificationDataset, self).append(
            image=image,
            label=label,
            category=category
        )
        self._lbp_images.append(kwargs['lbp_image'])
        self._lbp_hist.append(kwargs['lbp_hist'])
        self._feature_vec.append(kwargs['feature_vec'])

    def save_lbp_images(self, output_dir: str, fmt: str = 'jpg'):
        if f'.{fmt}' not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid image format {fmt}")
        output_dir = Path(output_dir)
        for i, (image, label, category, lbp_image, lbp_vector) in enumerate(self):
            if not Path(output_dir / category).exists():
                Path(output_dir / category).mkdir(parents=True, exist_ok=True)
            _cv2.imwrite(
                str(output_dir / category / f"{label}-{i}.lbp.{fmt}"),
                lbp_image
            )

    def _load_images(self, root: str, *, limit: int = 0, channel_flatten: bool = False):
        self._with_channel_flatten = channel_flatten
        categories = [d.name for d in Path(root).iterdir() if d.is_dir()]
        num_classes = len(categories)
        _LOGGER.info(f"Found {num_classes} categories: {categories}, loading images...")
        for i, c in enumerate(categories):
            path = Path(root) / c
            for j, f in enumerate(path.glob('*')):
                if f.suffix in _IMAGE_EXTENSIONS:
                    img = _cv2.imread(str(f))
                    if channel_flatten:
                        # Flatten the channel by stacking the RGB channel horizontally into an image
                        # with 3 times the width of the original image
                        image = _cv2.imread(str(f))
                        R, G, B = _cv2.split(image)
                        gray_img = _np.hstack((R, G, B))
                    else:
                        gray_img = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
                    lbp_img = self.faster_calculate_lbp(gray_img)
                    lbp_hist = self.calculate_lbp_hist(lbp_img)
                    feature_vec = _np.concatenate([lbp_hist, self.calculate_rgb_histogram(img)])
                    self.append(
                        image=img,
                        label=i,
                        category=c,
                        lbp_image=lbp_img,
                        lbp_hist=lbp_hist,
                        feature_vec=feature_vec
                    )
                    if limit != 0 and j >= limit / num_classes:
                        break
                else:
                    _LOGGER.warning(f"-- Skipping {f.name} due to unsupported format")

    def overview(self, save_path: Union[str, Path] = 'overview.png') -> None:
        figure = _plt.figure(figsize=(10, 12), dpi=300)
        gs = _gd.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, .8])

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

        ax2 = figure.add_subplot(gs[2])
        ax3 = figure.add_subplot(gs[3])

        # Display 9 random LBP images and their histogram
        inner_gs_ax2 = _gd.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[2], wspace=0.1, hspace=0.1)
        inner_gs_ax3 = _gd.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[3], wspace=0.1, hspace=0.1)
        for i in range(9):
            ax_ax2 = figure.add_subplot(inner_gs_ax2[i])
            ax_ax3 = figure.add_subplot(inner_gs_ax3[i])
            img_index = _np.random.randint(len(self._lbp_images))

            ax_ax2.imshow(self._lbp_images[img_index], cmap='gray')
            ax_ax2.set_title(f'Class: {self._categories[self._labels[img_index]]}', fontsize=10)
            ax_ax2.axis('off')

            ax_ax3.bar(_np.arange(256), self._lbp_hist[img_index], color='black')
            ax_ax3.set_title(f'Class: {self._categories[self._labels[img_index]]}', fontsize=10)
            ax_ax3.set_xticks([])
            ax_ax3.set_yticks([])

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.axis('off')

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.axis('off')

        _plt.tight_layout()
        _plt.show()
        _plt.savefig(save_path.__str__())

    @staticmethod
    def calculate_lbp(image: _np.ndarray) -> _np.ndarray:
        warnings.warn("Deprecated. Use `faster_calculate_lbp` instead.", DeprecationWarning)
        _LOGGER.warning("Using deprecated method `calculate_lbp`. Use `faster_calculate_lbp` instead.")
        lbp_image = _np.zeros_like(image)
        for i in range(1, lbp_image.shape[0] - 1):
            for j in range(1, lbp_image.shape[1] - 1):
                bin_str = ''
                center_value = image[i, j]
                bin_str += '1' if image[i - 1, j - 1] >= center_value else '0'
                bin_str += '1' if image[i - 1, j] >= center_value else '0'
                bin_str += '1' if image[i - 1, j + 1] >= center_value else '0'
                bin_str += '1' if image[i, j + 1] >= center_value else '0'
                bin_str += '1' if image[i + 1, j + 1] >= center_value else '0'
                bin_str += '1' if image[i + 1, j] >= center_value else '0'
                bin_str += '1' if image[i + 1, j - 1] >= center_value else '0'
                bin_str += '1' if image[i, j - 1] >= center_value else '0'

                lbp_image[i, j] = int(bin_str, 2)

        return lbp_image

    @staticmethod
    def faster_calculate_lbp(gray: _np.ndarray) -> _np.ndarray:
        """
        Calculate the Local Binary Pattern (LBP) representation of a grayscale image.

        This function computes the LBP representation of a grayscale image using the 8-neighborhood
        method. The LBP value of each pixel is computed by comparing its intensity value to the
        intensity values of its 8 neighbors. If the intensity of a neighbor is greater than or equal
        to the center pixel, a 1 is assigned to that neighbor, otherwise a 0 is assigned. The binary
        values of the neighbors are then concatenated in a clockwise order to form the LBP value of
        the center pixel.

        Args:
            gray (np.ndarray): The input grayscale image. Must be a 2D numpy array.

        Returns:
            np.ndarray: The LBP image, with the same shape as the input image.
        """
        padded_image = _np.pad(gray, pad_width=1, mode='edge')
        lbp_image = _np.zeros_like(gray, dtype=_np.int32)

        # Define the offsets for the 8 neighbors
        offsets = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if (i, j) != (0, 0)]

        for idx, (di, dj) in enumerate(offsets):
            # Shift the padded image using the offsets
            shifted_image = padded_image[1 + di: 1 + di + gray.shape[0], 1 + dj: 1 + dj + gray.shape[1]]
            # Update the binary representation of the LBP image
            lbp_image += (shifted_image >= padded_image[1:-1, 1:-1]) << idx

        return lbp_image.astype(_np.uint8)

    @staticmethod
    def calculate_lbp_hist(lbp_image: _np.ndarray) -> _np.ndarray:
        """
        Calculate the LBP histogram of an LBP image.

        This function computes the histogram of the LBP image, which represents the distribution of
        LBP values in the image. The histogram is normalized so that the sum of all bin values is 1.

        Args:
            lbp_image (np.ndarray): The input LBP image. Must be a 2D numpy array.

        Returns:
            np.ndarray: The LBP histogram, a 1D numpy array of shape (256,).
        """
        hist, _ = _np.histogram(lbp_image.ravel(), bins=_np.arange(0, 256 + 1), range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-06)
        return hist

    def export_csv(
        self,
        output_dir: Union[str, Path],
        train_test_split: bool = False,
        train_ratio: float = 0.8
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if train_test_split:
            train_csv = output_dir / "train.csv"
            train_csv.unlink() if train_csv.exists() else None
            test_csv = output_dir / "test.csv"
            test_csv.unlink() if test_csv.exists() else None
            train_size = int(len(self) * train_ratio)
            if train_size == 0 or train_size >= len(self):
                raise ValueError("Invalid train ratio")
            train_indices = _np.random.choice(len(self), train_size, replace=False)
            if isinstance(train_indices, int):
                train_indices = [train_indices]
            test_indices = _np.setdiff1d(_np.arange(len(self)), train_indices)
            self._export_csv(self[train_indices], train_csv)  # type: ignore
            self._export_csv(self[test_indices], test_csv)  # type: ignore

        else:
            output_csv = output_dir / "dataset.csv"
            output_csv.unlink() if output_csv.exists() else None
            self._export_csv(self, output_csv)

    @staticmethod
    def _export_csv(
        dataset: Iterable[Tuple[_np.ndarray, int, str, _np.ndarray, _np.ndarray]],
        output_csv: Path
    ):
        with open(output_csv, 'w') as f:
            for idx, (image, label, category, rgb_hist, lbp_image, lbp_hist, feature_vec) in enumerate(dataset):
                f.write(f"{label},")
                f.write(','.join(map(str, feature_vec)))
                f.write("\n")

    def reduce_lbp_dimensionality(self, method: str, **kwargs: Any) -> None:
        if method == 'pca':
            self._lbp_hist = _utils.pca_transform(self._lbp_hist, **kwargs)
        elif method == 'tsne':
            self._lbp_hist = _utils.tsne_transform(self._lbp_hist, **kwargs)
        else:
            raise NotImplementedError(f"Dimensionality reduction method {method} not implemented")

    def reduce_feature_dimensionality(self, method: str, **kwargs: Any) -> None:
        if method == 'pca':
            self._feature_vec = _utils.pca_transform(self._feature_vec, **kwargs)
        elif method == 'tsne':
            self._feature_vec = _utils.tsne_transform(self._feature_vec, **kwargs)
        else:
            raise NotImplementedError(f"Dimensionality reduction method {method} not implemented")

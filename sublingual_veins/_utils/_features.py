from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_transform(vec: List[np.ndarray], n_components: int) -> List[np.ndarray]:
    """
    Perform PCA transformation on a list of vectors.

    This function performs PCA transformation on a list of vectors and returns the transformed vectors.

    Args:
        vec (List[np.ndarray]): The list of vectors to transform.
        n_components (int): The number of principal components to keep.

    Returns:
        List[np.ndarray]: The transformed vectors.
    """
    pca = PCA(n_components=n_components)
    results = pca.fit_transform(vec)
    return [result for result in results]


def tsne_transform(vec: List[np.ndarray], n_components: int) -> List[np.ndarray]:
    """
    Perform t-SNE transformation on a list of vectors.

    This function performs t-SNE transformation on a list of vectors and returns the transformed vectors.

    Args:
        vec (List[np.ndarray]): The list of vectors to transform.
        n_components (int): The number of dimensions to reduce to.

    Returns:
        List[np.ndarray]: The transformed vectors.
    """

    tsne = TSNE(n_components=n_components)
    results = tsne.fit_transform(vec)
    return [result for result in results]

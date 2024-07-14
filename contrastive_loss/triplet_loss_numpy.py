import numpy as np


def triplet_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the triplet loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        anchor (np.ndarray): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (np.ndarray): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (np.ndarray): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The triplet loss
    """
    distance_ap = np.sqrt(np.sum(np.square(anchor - positive), axis=1))
    distance_an = np.sqrt(np.sum(np.square(anchor - negative), axis=1))

    tr_loss = np.maximum(0, distance_ap - distance_an + margin)
    loss = np.mean(tr_loss)
    return loss

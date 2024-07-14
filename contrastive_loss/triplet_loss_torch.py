import torch


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 5.0,
) -> torch.Tensor:
    """
    Computes the triplet loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        anchor (torch.Tensor): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (torch.Tensor): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (torch.Tensor): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The triplet loss
    """
    distance_ap = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1))
    distance_an = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1))
    tr_loss = torch.maximum(torch.zeros(distance_ap.shape),
                           distance_ap - distance_an + margin)
    loss = torch.mean(tr_loss)
    return loss

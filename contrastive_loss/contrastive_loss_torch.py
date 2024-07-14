import torch


def contrastive_loss(
    x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, margin: float = 5.0
) -> torch.Tensor:
    """
    Computes the contrastive loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        x1 (torch.Tensor): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (torch.Tensor): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (torch.Tensor): Ground truth labels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The contrastive loss
    """
    distance = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))
    contr_loss = y * (distance ** 2) + (1 - y) * \
                (torch.maximum(margin - distance, torch.zeros(distance.shape)) ** 2)
    loss = torch.mean(contr_loss)
    return loss

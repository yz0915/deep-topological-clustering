import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein_distance(embeddings1, embeddings2):
    """
    Compute the Wasserstein distance between two sets of graph embeddings.

    Parameters:
    - embeddings1 (np.array): Node embeddings for the first graph. Shape (n1, d) where n1 is the number of nodes and d is the dimension of the embeddings.
    - embeddings2 (np.array): Node embeddings for the second graph. Shape (n2, d) where n2 is the number of nodes and d is the dimension of the embeddings.

    Returns:
    - float: The average Wasserstein distance across all dimensions of the embeddings.
    """
    if embeddings1.shape[1] != embeddings2.shape[1]:
        raise ValueError("Embeddings must be of the same dimensionality")

    distances = []
    for dim in range(embeddings1.shape[1]):
        distance = wasserstein_distance(embeddings1[:, dim], embeddings2[:, dim])
        distances.append(distance)

    return np.mean(distances)


# Assume embeddings1 and embeddings2 are your node embeddings from the original and autoencoded graphs respectively
embeddings1 = np.random.rand(10, 5)  # 10 nodes, 5-dimensional embeddings
embeddings2 = np.random.rand(8, 5)  # 8 nodes, 5-dimensional embeddings

w_distance = compute_wasserstein_distance(embeddings1, embeddings2)
print("Wasserstein Distance:", w_distance)

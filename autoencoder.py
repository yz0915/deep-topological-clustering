import sys
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.cluster import contingency_matrix
from matplotlib import pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


class GNN(torch.nn.Module):
    def __init__(self, num_nodes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_nodes, num_nodes // 2)
        self.conv2 = GCNConv(num_nodes // 2, num_nodes)

    def forward(self, x, batch):
        edge_index = torch.nonzero(x, as_tuple=False).t()
        # print(f"Edge index shape: {edge_index.shape}")
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # Global mean pooling
        x = global_mean_pool(x, batch)  # batch is the index of the batch to which the nodes belong
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class TopClustering:
    """Topological clustering.
    
    Attributes:
        n_clusters: 
          The number of clusters.
        top_relative_weight:
          Relative weight between the geometric and topological terms.
          A floating point number between 0 and 1.
        max_iter_alt:
          Maximum number of iterations for the topological clustering.
        max_iter_interp:
          Maximum number of iterations for the topological interpolation.
        learning_rate:
          Learning rate for the topological interpolation.
        
    Reference:
        Songdechakraiwut, Tananun, Bryan M. Krause, Matthew I. Banks, Kirill V. Nourski, and Barry D. Van Veen. 
        "Fast topological clustering with Wasserstein distance." 
        International Conference on Learning Representations (ICLR). 2022.
    """

    def __init__(self, n_clusters, top_relative_weight, max_iter_alt,
                 max_iter_interp, learning_rate):
        self.n_clusters = n_clusters
        self.top_relative_weight = top_relative_weight
        self.max_iter_alt = max_iter_alt
        self.max_iter_interp = max_iter_interp
        self.learning_rate = learning_rate

    def fit_predict(self, dataset, batch):
        n_node = dataset[0].shape[0]  # Assuming all graphs have the same number of nodes

        encoder = GNN(n_node)
        decoder = MLP(input_dim=60, output_dim=3600)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
        criterion = torch.nn.MSELoss()

        for epoch in range(200):
            encoder.train()
            decoder.train()
            total_loss = 0

            for i, adj_matrix in enumerate(dataset):
                x = torch.tensor(adj_matrix, dtype=torch.float32)
                # compute mst and nonmst for original graph
                # ori_mst, ori_nonmst = self._compute_birth_death_sets(adj_matrix)
                # ori_mst = torch.tensor(ori_mst, dtype=torch.float32)
                # ori_nonmst = torch.tensor(ori_nonmst, dtype=torch.float32)
                mst_index, nonmst_index = self._compute_birth_death_sets(adj_matrix)
                ori_mst = torch.take(x, mst_index)
                ori_nonmst = torch.take(x, nonmst_index)

                b = batch[i*60:(i+1)*60]
                b = torch.tensor(b, dtype=torch.long)

                encoded = encoder(x, b)
                decoded = decoder(encoded)
                # print(f"Shape of decoded_adj before reshape: {decoded_adj.shape}")
                
                # Assuming decoded_adj is a tensor of shape [batch_size, 3600]
                # Reshape decoded_adj to be a 60x60 matrix
                # batch_size = decoded_adj.shape[0]
                # print(decoded)
                decoded = decoded[-1]
                decoded = decoded.view(n_node, n_node)

                # compute mst and nonmst for new graph
                # Use detach() when converting to numpy for non-gradient requiring operations
                # decoded = decoded.detach().numpy()
                # new_mst, new_nonmst = self._compute_birth_death_sets(decoded)
                # new_mst = torch.tensor(new_mst, dtype=torch.float32, requires_grad=True)
                # new_nonmst = torch.tensor(new_nonmst, dtype=torch.float32, requires_grad=True)
                # new_mst_index, new_nonmst_index = self._compute_birth_death_sets(decoded)
                new_mst = torch.take(decoded, mst_index)
                new_nonmst = torch.take(decoded, nonmst_index)


                 # Compute MSE losses
                loss_mst = criterion(new_mst, ori_mst)
                loss_nonmst = criterion(new_nonmst, ori_nonmst)

                # Sum the losses
                loss = loss_mst + loss_nonmst
                loss.backward()  # Backpropagate errors immediately
                total_loss += loss.item()

            optimizer.step()  # Update parameters(Apply gradient updates) once for the entire batch
            optimizer.zero_grad() # Reset gradients after update

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {total_loss / len(dataset)}")

        return 


    def _vectorize_geo_top_info(self, adj):
        birth_set, death_set = self._compute_birth_death_sets(
            adj)  # topological info
        vec = adj[np.triu_indices(adj.shape[0], k=1)]  # geometric info
        return np.concatenate((vec, birth_set, death_set), axis=0)

    # def _compute_birth_death_sets(self, adj):
    #     """Computes birth and death sets of a network."""
    #     mst, nonmst = self._bd_demomposition(adj)
    #     birth_ind = np.nonzero(mst)
    #     death_ind = np.nonzero(nonmst)
    #     return np.sort(mst[birth_ind]), np.sort(nonmst[death_ind])

    def _bd_demomposition(self, adj):
        """Birth-death decomposition."""
        eps = np.nextafter(0, 1)
        adj[adj == 0] = eps
        adj = np.triu(adj, k=1)
        Xcsr = csr_matrix(-adj)
        Tcsr = minimum_spanning_tree(Xcsr)
        mst = -Tcsr.toarray()  # reverse the negative sign
        nonmst = adj - mst
        return mst, nonmst

    def _compute_birth_death_sets(self, adj):
        mst, nonmst = self._bd_demomposition(adj)
        n = adj.shape[0]
        birth_ind = np.nonzero(mst)
        death_ind = np.nonzero(nonmst)

        # Convert 2D indices to 1D and get corresponding values
        mst_flat_indices = np.ravel_multi_index(birth_ind, (n, n))
        nonmst_flat_indices = np.ravel_multi_index(death_ind, (n, n))

        # Get values from the original adjacency matrix using these flat indices
        mst_values = adj.flatten()[mst_flat_indices]
        nonmst_values = adj.flatten()[nonmst_flat_indices]

        # Sort indices by these values
        sorted_mst_indices = mst_flat_indices[np.argsort(mst_values)]
        sorted_nonmst_indices = nonmst_flat_indices[np.argsort(nonmst_values)]

        return torch.from_numpy(sorted_mst_indices), torch.from_numpy(sorted_nonmst_indices)
        

    def _get_nearest_centroid(self, X, centroids):
        """Determines cluster membership of data points."""
        dist = self._compute_top_dist(X, centroids)
        nearest_centroid_index = np.argmin(dist, axis=1)
        return nearest_centroid_index

    def _compute_top_dist(self, X, centroid):
        """Computes the pairwise top. distances between networks and centroids."""
        return np.dot((X - centroid)**2, self.weight_array)

    def _top_interpolation(self, init_centroid, sample_mean,
                           top_centroid_birth_set, top_centroid_death_set):
        """Topological interpolation."""
        curr = init_centroid
        for _ in range(self.max_iter_interp):
            # Geometric term gradient
            geo_gradient = 2 * (curr - sample_mean)

            # Topological term gradient
            sorted_birth_ind, sorted_death_ind = self._compute_optimal_matching(
                curr)
            top_gradient = np.zeros_like(curr)
            top_gradient[sorted_birth_ind] = top_centroid_birth_set
            top_gradient[sorted_death_ind] = top_centroid_death_set
            top_gradient = 2 * (curr - top_gradient)

            # Gradient update
            curr -= self.learning_rate * (
                (1 - self.top_relative_weight) * geo_gradient +
                self.top_relative_weight * top_gradient)
        return curr

    def _compute_optimal_matching(self, adj):
        mst, nonmst = self._bd_demomposition(adj)
        birth_ind = np.nonzero(mst)
        death_ind = np.nonzero(nonmst)
        sorted_temp_ind = np.argsort(mst[birth_ind])
        sorted_birth_ind = tuple(np.array(birth_ind)[:, sorted_temp_ind])
        sorted_temp_ind = np.argsort(nonmst[death_ind])
        sorted_death_ind = tuple(np.array(death_ind)[:, sorted_temp_ind])
        return sorted_birth_ind, sorted_death_ind


#############################################
################### Demo ####################
#############################################
def random_modular_graph(d, c, p, mu, sigma):
    """Simulated modular network.
    
        Args:
            d: Number of nodes.
            c: Number of clusters/modules.
            p: Probability of attachment within module.
            mu, sigma: Used for random edge weights.
            
        Returns:
            Adjacency matrix.
    """
    adj = np.zeros((d, d))  # adjacency matrix
    for i in range(1, d + 1):
        for j in range(i + 1, d + 1):
            module_i = math.ceil(c * i / d)
            module_j = math.ceil(c * j / d)

            # Within module
            if module_i == module_j:
                if random.uniform(0, 1) <= p:
                    w = np.random.normal(mu, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
                else:
                    w = np.random.normal(0, sigma)
                    adj[i - 1][j - 1] = max(w, 0)

            # Between modules
            else:
                if random.uniform(0, 1) <= 1 - p:
                    w = np.random.normal(mu, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
                else:
                    w = np.random.normal(0, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
    return adj


def purity_score(labels_true, labels_pred):
    mtx = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(mtx, axis=0)) / np.sum(mtx)


def main():
    np.random.seed(0)
    random.seed(0)

    # Generate a dataset comprising simulated modular networks
    dataset = []
    labels_true = []
    n_network = 20
    n_node = 60
    p = 0.7
    mu = 1
    sigma = 0.5
    batch = []

    for module_index, module in enumerate([2, 3, 5]):
        for graph_index in range(n_network):
            adj = random_modular_graph(n_node, module, p, mu, sigma)
            dataset.append(adj)
            labels_true.append(module)
            batch.extend([module_index * n_network + graph_index] * n_node)

    # Topological clustering
    n_clusters = 3
    top_relative_weight = 0.99  # 'top_relative_weight' between 0 and 1
    max_iter_alt = 300
    max_iter_interp = 300
    learning_rate = 0.05
    print('Topological clustering\n----------------------')
    r = TopClustering(n_clusters, top_relative_weight, max_iter_alt,
                                max_iter_interp,
                                learning_rate).fit_predict(dataset, batch)
    
    # Flattening adjacency matrices for potential silhouette score calculation
    flattened_dataset = np.array([np.ravel(adj) for adj in dataset])

    # Performance Evaluation
    ari = adjusted_rand_score(labels_true, r)
    nmi = normalized_mutual_info_score(labels_true, r)
    # silhouette = silhouette_score(flattened_dataset, r) if len(set(r)) > 1 else 0

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    # print("Silhouette Score:", silhouette)


if __name__ == '__main__':
    main()

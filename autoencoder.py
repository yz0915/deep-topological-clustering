import sys
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.cluster import contingency_matrix
from matplotlib import pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans


class GNN(torch.nn.Module):
    def __init__(self, num_nodes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_nodes, num_nodes // 2)
        self.conv2 = GCNConv(num_nodes // 2, num_nodes)

    def forward(self, x, batch):
        edge_index = torch.nonzero(x, as_tuple=False).t()
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

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

class DeepKMeans(nn.Module):
    def __init__(self, d_embed, n_clusters, lambd):

        super(DeepKMeans, self).__init__()

        self.d_embed = d_embed
        self.k = n_clusters
        self.lambd = lambd

        self.centroids = nn.Parameter(torch.FloatTensor(n_clusters, d_embed).uniform_(-1, 1))

    def forward(self, x, alpha):
        # Note that x is the graph embedding from some autoencoder
        list_dist = []
        for i in range(self.centroids.size(0)):
            dist = self.get_dist(x, self.centroids[i].unsqueeze(0))
            list_dist.append(dist)
        stack_dist = torch.stack(list_dist)

        min_dist = torch.min(stack_dist, dim=0)[0]

        list_exp = []
        for i in range(self.centroids.size(0)):
            exp = torch.exp(-alpha * (stack_dist[i] - min_dist))
            list_exp.append(exp)
        stack_exp = torch.stack(list_exp)
        sum_exponentials = torch.sum(stack_exp, dim=0)

        list_softmax = []
        list_weighted_dist = []
        for i in range(self.centroids.size(0)):
            softmax = stack_exp[i] / sum_exponentials
            weighted_dist = stack_dist[i] * softmax
            list_softmax.append(softmax)
            list_weighted_dist.append(weighted_dist)
        stack_weighted_dist = torch.stack(list_weighted_dist)

        # kmeans_loss = torch.mean(torch.sum(stack_weighted_dist, dim=0))
        # loss = self.lambd * kmeans_loss

        return stack_weighted_dist

    def get_dist(self, x, y):
        return torch.sum((x - y) ** 2, dim=1)

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
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=self.max_iter_alt)

    # def autoencoder(self, dataset, batch):
    #     n_node = dataset[0].shape[0]  # Assuming all graphs have the same number of nodes
    #     encoder = GNN(n_node)
    #     decoder = MLP(input_dim=60, output_dim=3600)
    #     optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    #     criterion = torch.nn.MSELoss()

    #     gradient_norms = []  # List to store gradient norms
    #     embeddings = []  # List to store embeddings

    #     # Pretraining Phase
    #     for epoch in range(200):
    #         encoder.train()
    #         decoder.train()
    #         total_loss = 0

    #         for i, adj_matrix in enumerate(dataset):
    #             x = torch.tensor(adj_matrix, dtype=torch.float32)
    #             mst_index, nonmst_index = self._compute_birth_death_sets(adj_matrix)
    #             ori_mst = torch.take(x, mst_index)
    #             ori_nonmst = torch.take(x, nonmst_index)

    #             # b = batch[i*60:(i+1)*60]
    #             # b = torch.tensor(b, dtype=torch.long)
    #             b = torch.zeros(60, dtype=torch.long)
    #             encoded = encoder(x, b)
    #             if epoch == 199:  # Only save embeddings in the last epoch
    #                 embeddings.append(torch.squeeze(encoded).detach().numpy())  # Save the embeddings to the list

    #             decoded = decoder(encoded)
    #             decoded = decoded.view(n_node, n_node) # Reshape decoded_adj to be a 60x60 matrix

    #             new_mst = torch.take(decoded, mst_index)
    #             new_nonmst = torch.take(decoded, nonmst_index)

    #              # Compute MSE losses
    #             loss_mst = criterion(new_mst, ori_mst)
    #             loss_nonmst = criterion(new_nonmst, ori_nonmst)

    #             # Sum the losses
    #             loss = loss_mst + loss_nonmst
    #             loss.backward()  # Backpropagate errors immediately
    #             total_loss += loss.item()

    #         # Get the gradient norm
    #         grad_norm = encoder.conv1.lin.weight.grad.norm().item()
    #         gradient_norms.append(grad_norm)

    #         optimizer.step()  # Update parameters(Apply gradient updates) once for the entire batch
    #         optimizer.zero_grad() # Reset gradients after update

    #         if epoch % 10 == 0:
    #             print(f"Epoch {epoch}, Average Loss: {total_loss / len(dataset)}")

    #     plt.plot(gradient_norms)
    #     plt.title('Gradient Norms of conv1 Weights Over Epochs')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Gradient Norm')
    #     plt.grid(True)
    #     plt.show()
    #     # return gradient_norms
    #     return embeddings

    def autoencoder(self, dataset, batch):
        n_node = dataset[0].shape[0]  # Assuming all graphs have the same number of nodes
        encoder = GNN(n_node)
        decoder = MLP(input_dim=60, output_dim=3600)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
        criterion = torch.nn.MSELoss()

        gradient_norms = []  # List to store gradient norms
        embeddings = []  # List to store embeddings

        # Pretraining Phase
        for epoch in range(200):
            encoder.train()
            decoder.train()
            total_loss = 0

            for i, adj_matrix in enumerate(dataset):
                x = torch.tensor(adj_matrix, dtype=torch.float32)
                mst_index, nonmst_index = self._compute_birth_death_sets(adj_matrix)
                ori_mst = torch.take(x, mst_index)
                ori_nonmst = torch.take(x, nonmst_index)

                b = torch.zeros(60, dtype=torch.long)
                encoded = encoder(x, b)
                if epoch == 199:  # Only save embeddings in the last epoch
                    embeddings.append(torch.squeeze(encoded).detach().numpy())  # Save the embeddings to the list
                decoded = decoder(encoded)
                decoded = decoded.view(n_node, n_node) # Reshape decoded_adj to be a 60x60 matrix

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

        # Initialize KMeans centroids
        self.kmeans.fit(embeddings)
        for epoch in range(100): 
            total_loss = 0
            for i, adj_matrix in enumerate(dataset):
                x = torch.tensor(adj_matrix, dtype=torch.float32)
                mst_index, nonmst_index = self._compute_birth_death_sets(adj_matrix)
                ori_mst = torch.take(x, mst_index)
                ori_nonmst = torch.take(x, nonmst_index)

                b = torch.zeros(60, dtype=torch.long)
                encoded = encoder(x, b)
                decoded = decoder(encoded)
                decoded = decoded.view(n_node, n_node) # Reshape decoded_adj to be a 60x60 matrix

                new_mst = torch.take(decoded, mst_index)
                new_nonmst = torch.take(decoded, nonmst_index)

                # Compute MSE losses
                loss_mst = criterion(new_mst, ori_mst)
                loss_nonmst = criterion(new_nonmst, ori_nonmst)

                # Calculate KMeans loss
                closest_centers = self.kmeans.predict(embeddings)
                loss_kmeans = criterion(embeddings, closest_centers)
                # loss_kmeans = np.mean(np.sum((embeddings - closest_centers) ** 2, axis=1))
                # loss_kmeans = torch.tensor(loss_kmeans, requires_grad=True)

                total_loss = loss_mst + loss_nonmst + loss_kmeans
                total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item()}")


    def dkm(self, embeddings):
        # Fit the KMeans model only if centroids are not already determined.
        if self.kmeans.cluster_centers_ is None:
            self.kmeans.fit(embeddings)  # Fit KMeans to obtain initial centroids.
        
        # Predict using the existing centroids, without refitting.
        return self.kmeans.predict(embeddings) # return the array of the cluster labels for each embedding
    
    # def fit_predict(self, data):
    #     """Computes topological clustering and predicts cluster index for each sample.
        
    #         Args:
    #             data:
    #               Training instances to cluster.
                  
    #         Returns:
    #             Cluster index each sample belongs to.
    #     """
    #     data = np.asarray(data)
    #     print(data.shape)
    #     n_node = data.shape[-1]
    #     n_edges = math.factorial(n_node) // math.factorial(2) // math.factorial(
    #         n_node - 2)  # n_edges = (n_node choose 2)
    #     n_births = n_node - 1
    #     self.weight_array = np.append(
    #         np.repeat(1 - self.top_relative_weight, n_edges),
    #         np.repeat(self.top_relative_weight, n_edges))

    #     # Networks represented as vectors concatenating geometric and topological info
    #     X = data
    #     # for adj in data:
    #     #     X.append(self._vectorize_geo_top_info(adj))
    #     # X = np.asarray(X)

    #     # Random initial condition
    #     self.centroids = X[random.sample(range(X.shape[0]), self.n_clusters)]

    #     # Assign the nearest centroid index to each data point
    #     assigned_centroids = self._get_nearest_centroid(
    #         X[:, None, :], self.centroids[None, :, :])
    #     prev_assigned_centroids = assigned_centroids

    #     for it in range(self.max_iter_alt):
    #         for cluster in range(self.n_clusters):
    #             # Previous iteration centroid
    #             prev_centroid = np.zeros((n_node, n_node))
    #             prev_centroid[np.triu_indices(
    #                 prev_centroid.shape[0],
    #                 k=1)] = self.centroids[cluster][:n_edges]

    #             # Determine data points belonging to each cluster
    #             cluster_members = X[assigned_centroids == cluster]

    #             # Compute the sample mean and top. centroid of the cluster
    #             cc = cluster_members.mean(axis=0)
    #             sample_mean = np.zeros((n_node, n_node))
    #             sample_mean[np.triu_indices(sample_mean.shape[0],
    #                                         k=1)] = cc[:n_edges]
    #             top_centroid = cc[n_edges:]
    #             top_centroid_birth_set = top_centroid[:n_births]
    #             top_centroid_death_set = top_centroid[n_births:]

    #             # Update the centroid
    #             try:
    #                 cluster_centroid = self._top_interpolation(
    #                     prev_centroid, sample_mean, top_centroid_birth_set,
    #                     top_centroid_death_set)
    #                 self.centroids[cluster] = self._vectorize_geo_top_info(
    #                     cluster_centroid)
    #             except:
    #                 print(
    #                     'Error: Possibly due to the learning rate is not within appropriate range.'
    #                 )
    #                 sys.exit(1)

    #         # Update the cluster membership
    #         assigned_centroids = self._get_nearest_centroid(
    #             X[:, None, :], self.centroids[None, :, :])

    #         # Compute and print loss as it is progressively decreasing
    #         loss = self._compute_top_dist(
    #             X, self.centroids[assigned_centroids]).sum() / len(X)
    #         print('Iteration: %d -> Loss: %f' % (it, loss))

    #         if (prev_assigned_centroids == assigned_centroids).all():
    #             break
    #         else:
    #             prev_assigned_centroids = assigned_centroids
    #     return assigned_centroids

    def _vectorize_geo_top_info(self, adj):
        birth_set, death_set = self._compute_birth_death_sets(
            adj)  # topological info
        vec = adj[np.triu_indices(adj.shape[0], k=1)]  # geometric info
        return np.concatenate((vec, birth_set, death_set), axis=0)

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

    embeddings = TopClustering(n_clusters, top_relative_weight, max_iter_alt,
                            max_iter_interp,
                            learning_rate).autoencoder(dataset, batch)
    
    print('Topological clustering\n----------------------')
    labels_pred = TopClustering(n_clusters, top_relative_weight, max_iter_alt,
                                max_iter_interp,
                                learning_rate).dkm(embeddings)
    print('\nResults\n-------')
    print('True labels:', np.asarray(labels_true))
    print('Pred indices:', labels_pred)
    print('Purity score:', purity_score(labels_true, labels_pred))


if __name__ == '__main__':
    main()

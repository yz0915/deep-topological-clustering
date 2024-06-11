
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
    
class GraphKMeans(nn.Module):

    # Note that we expect num_nodes = d_embed
    def __init__(self, num_nodes, d_embed, n_clusters, lambd, initial_clusters):

        super(GraphKMeans, self).__init__()

        self.num_nodes = num_nodes
        self.d_embed = d_embed
        self.k = n_clusters
        self.lambd = lambd
        
        self.pretrain = pretrain
        self.centroids = nn.Parameter(torch.FloatTensor(initial_clusters))

    def forward(self, embeddings, alpha):
        
        # Note that x is the graph embedding from some autoencoder
        list_dist = []
        for i in range(self.centroids.size(0)):
            dist = self.f_dist(embeddings, self.centroids[i].unsqueeze(0))
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

        k_means_loss = self.lambd * torch.mean(torch.sum(stack_weighted_dist, dim=0))
        return k_means_loss

    def f_dist(self, x, y):
        return torch.sum((x - y) ** 2, dim=1)

def pretrain(dataset, epochs=100):

    print("PRETRAINING")

    # Pretrain an encoder via autoencoder reconstruction loss
    num_nodes = dataset[0].shape[0]

    encoder = GNN(num_nodes)
    decoder = MLP(num_nodes, num_nodes**2)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    criterion = torch.nn.MSELoss()

    final_embeddings = []

    for epoch in range(epochs):

        encoder.train()
        decoder.train()
        total_loss = 0

        for i, adj_matrix in enumerate(dataset):

            x = torch.tensor(adj_matrix, dtype=torch.float32)
            mst_index, nonmst_index = _compute_birth_death_sets(adj_matrix)
            ori_mst = torch.take(x, mst_index)
            ori_nonmst = torch.take(x, nonmst_index)

            b = torch.zeros(60, dtype=torch.long)
            encoded = encoder(x, b)
            
            if epoch == epochs-1:
                final_embeddings.append(torch.squeeze(encoded).detach().numpy())

            decoded = decoder(encoded)
            decoded = decoded.view(num_nodes, num_nodes) # Reshape decoded_adj to be a 60x60 matrix

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

    return encoder, decoder, final_embeddings

def train(dataset, num_clusters=3, epochs=100):

    num_nodes = dataset[0].shape[0]
    encoder, decoder, pretrained_embeddings = pretrain(dataset)

    # Run k-means++ to get initial cluster distribution
    kmeans_model = KMeans(n_clusters=num_clusters, init="k-means++").fit(pretrained_embeddings)
    deep_k_means = GraphKMeans(num_nodes, num_nodes, num_clusters, 0.1, kmeans_model.cluster_centers_)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(deep_k_means.parameters()), lr=0.01)
    criterion = torch.nn.MSELoss()

    print("TRAINING")

    for epoch in range(epochs):

        loss = 0

        total_loss = 0
        deep_k_means.train()

        embeddings = []  # List to store embeddings

        for i, adj_matrix in enumerate(dataset):

            x = torch.tensor(adj_matrix, dtype=torch.float32)
            mst_index, nonmst_index = _compute_birth_death_sets(adj_matrix)
            ori_mst = torch.take(x, mst_index)
            ori_nonmst = torch.take(x, nonmst_index)

            b = torch.zeros(60, dtype=torch.long)
            encoded = encoder(x, b)
            embeddings.append(encoded)
            
            decoded = decoder(encoded)
            decoded = decoded.view(num_nodes, num_nodes) # Reshape decoded_adj to be a 60x60 matrix

            new_mst = torch.take(decoded, mst_index)
            new_nonmst = torch.take(decoded, nonmst_index)

            # Compute MSE losses
            loss_mst = criterion(new_mst, ori_mst)
            loss_nonmst = criterion(new_nonmst, ori_nonmst)

            # Compute MSE losses
            loss_mst = criterion(new_mst, ori_mst)
            loss_nonmst = criterion(new_nonmst, ori_nonmst)
            loss = loss + loss_mst + loss_nonmst

        loss_k_means = deep_k_means(torch.stack(embeddings), 0.5)
        loss = loss + loss_k_means
        loss.backward()  # Backpropagate errors immediately
        total_loss += loss.item()

        optimizer.step()  # Update parameters(Apply gradient updates) once for the entire batch
        optimizer.zero_grad() # Reset gradients after update

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {total_loss / len(dataset)}")


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

def _bd_demomposition(adj):
    """Birth-death decomposition."""
    eps = np.nextafter(0, 1)
    adj[adj == 0] = eps
    adj = np.triu(adj, k=1)
    Xcsr = csr_matrix(-adj)
    Tcsr = minimum_spanning_tree(Xcsr)
    mst = -Tcsr.toarray()  # reverse the negative sign
    nonmst = adj - mst
    return mst, nonmst

def _compute_birth_death_sets(adj):
    mst, nonmst = _bd_demomposition(adj)
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

    train(dataset)

if __name__ == '__main__':
    main()
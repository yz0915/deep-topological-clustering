import sys
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp

import wandb

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.cluster import contingency_matrix
from matplotlib import pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.linalg import eigh

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, degree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, feat_dim, hidden_dim1):
        super(VAE, self).__init__()
        self.conv1 = GCNConv(feat_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = GCNConv(feat_dim, hidden_dim1)
        self.dropout2 = nn.Dropout(0.5)

    def encode(self, x, adj, batch):
        mu = self.conv1(x, adj).relu()
        mu_pooled = global_mean_pool(mu, batch)

        logvar = self.conv2(x, adj).relu()
        logvar_pooled = global_mean_pool(logvar, batch)

        return mu_pooled, logvar_pooled

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, adj, batch):
        mu_pooled, logvar_pooled = self.encode(x, adj, batch)
        z_pooled = self.reparameterize(mu_pooled, logvar_pooled)
        return z_pooled, mu_pooled, logvar_pooled

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(output_dim // 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.layers(x)

class AutoregressiveDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_steps):
        super(AutoregressiveDecoder, self).__init__()
        self.num_steps = num_steps  # Number of steps to generate (e.g., number of nodes or edges)
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, z):
        # Start with initial input, could be the latent vector repeated or transformed
        x = z.unsqueeze(1).repeat(1, self.num_steps, 1)  # Shape: [batch_size, num_steps, input_dim]
        
        # Initialize hidden state, optionally could be a function of z
        h0 = torch.zeros(1, z.size(0), self.hidden_dim, device=z.device)
        
        # Autoregressively generate outputs
        outputs, _ = self.rnn(x, h0)
        stepwise_outputs = self.output_layer(outputs)
        
        return stepwise_outputs
    
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

def pretrain(dataset, numSampledCCs, numSampledCycles, alpha, gamma, epochs, lr, latent_space_VAE, WASS_LOSS, Autoregressive=True):

    print("PRETRAINING")

    train_loader, adj_matrices = dataset

    feat_dim = train_loader.num_features

    max_nodes = max(data.num_nodes for data in train_loader)

    encoder = VAE(feat_dim, latent_space_VAE).to(device)
    if WASS_LOSS:
        decoder = MLP(latent_space_VAE, numSampledCCs + numSampledCycles).to(device)
    else:
        output_dim = max_nodes * (max_nodes - 1) // 2
        decoder = MLP(latent_space_VAE, output_dim).to(device)

    if Autoregressive:
        # Adjust dimensions according to your specific use case
        input_dim = latent_space_VAE  # Dimension of the latent space
        hidden_dim = 32  # Hidden dimension of the RNN
        output_dim = 1  # Dimension of output at each step, adjust based on your graph structure
        num_steps = max_nodes * (max_nodes - 1) // 2  # Number of nodes or edges to predict, define based on your graph's characteristics
        decoder = AutoregressiveDecoder(input_dim, hidden_dim, output_dim, num_steps).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()

    final_embeddings = []

    for epoch in range(epochs):

        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        cur_loss = 0

        for data, adj_matrix in zip(train_loader, adj_matrices):

            data = data.to(device)
            ori_adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

            if WASS_LOSS:
                ori_mst_index, ori_nonmst_index = _compute_birth_death_sets(adj_matrix, numSampledCCs, numSampledCycles)
                ori_mst = torch.take(ori_adj.to(device), ori_mst_index.to(device))
                ori_nonmst = torch.take(ori_adj.to(device), ori_nonmst_index.to(device))

                adj_norm = preprocess_graph(adj_matrix)
                z_pooled, mu_pooled, logvar_pooled = encoder(data.x, adj_norm, data.batch)
            
                if epoch == epochs-1:
                    final_embeddings.append(torch.squeeze(z_pooled).detach().cpu().numpy())

                decoded = decoder(z_pooled)

                # Reconstruct the adjacency matrix from the top triangle
                recons_adj = torch.zeros((numSampledCCs+1, numSampledCCs+1), dtype=torch.float32).to(device)
                upper_indices = torch.triu_indices(numSampledCCs+1, numSampledCCs+1, offset=1).to(device)
                recons_adj[upper_indices[0], upper_indices[1]] = decoded
                recons_adj = recons_adj + recons_adj.t()

                recons_adj_n = recons_adj.detach().cpu().numpy()
                new_mst_index, new_nonmst_index = _compute_birth_death_sets(recons_adj_n, numSampledCCs, numSampledCycles)

                new_mst = torch.take(recons_adj.to(device), new_mst_index.to(device))
                new_nonmst = torch.take(recons_adj.to(device), new_nonmst_index.to(device))

                # Compute MSE losses
                loss_mst = criterion(new_mst, ori_mst)
                loss_nonmst = criterion(new_nonmst, ori_nonmst)

                n_nodes = data.num_nodes
                loss_kl = kl_loss(mu_pooled, logvar_pooled, n_nodes)

                loss = alpha * loss_mst + alpha * loss_nonmst + gamma * loss_kl

                loss.backward()
                cur_loss += loss.item()

            else:
                
                adj_norm = preprocess_graph(adj_matrix)
                z_pooled, mu_pooled, logvar_pooled = encoder(data.x, adj_norm, data.batch)
            
                if epoch == epochs-1:
                    final_embeddings.append(torch.squeeze(z_pooled).detach().cpu().numpy())

                z_pooled = decoder(z_pooled)

                # Reconstruct the adjacency matrix from the top triangle
                max_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float32).to(device)
                upper_indices = torch.triu_indices(max_nodes, max_nodes, offset=1).to(device)
                # max_matrix[upper_indices[0], upper_indices[1]] = z_pooled
                max_matrix[upper_indices[0], upper_indices[1]] = z_pooled.squeeze()
                max_matrix = max_matrix + max_matrix.t()

                n_nodes = data.num_nodes
                reshape_adj = max_matrix.to(device).view(-1)[:n_nodes**2]
                reshape_adj = reshape_adj.view(n_nodes, n_nodes)

                loss_mse = criterion(reshape_adj, ori_adj)
                loss_kl = kl_loss(mu_pooled, logvar_pooled, n_nodes)

                loss = alpha * loss_mse + gamma * loss_kl

                loss.backward()
                cur_loss += loss.item()

        optimizer.step()  # Update parameters(Apply gradient updates) once for the entire batch

        # wandb.log({"pretrain_loss": cur_loss})
        if epoch % 10 == 0:
            # print(f"Epoch {epoch}, Average Loss: {cur_loss / len(train_loader)}")
            print(f"Epoch {epoch}, Loss: {cur_loss}")

    return encoder, decoder, final_embeddings

def train(dataset, hyperparameters, num_clusters=6, WASS_LOSS=False):

    train_loader, adj_matrices = dataset

    max_nodes = max(data.num_nodes for data in train_loader)

    epochs, pretrain_epochs, learning_rate, numSampledCCs, alpha, beta, gamma, latent_space_VAE = hyperparameters
    numSampledCycles = ((numSampledCCs+1) * numSampledCCs) // 2 - numSampledCCs
    # new_adj_dim = numSampledCCs + numSampledCycles

    encoder, decoder, pretrained_embeddings = pretrain(dataset, numSampledCCs=numSampledCCs,
                                                       numSampledCycles=numSampledCycles,
                                                       alpha=alpha, gamma=gamma,
                                                       epochs=pretrain_epochs,
                                                       lr=learning_rate,
                                                       latent_space_VAE = latent_space_VAE,
                                                       WASS_LOSS=WASS_LOSS)

    # Run k-means++ to get initial cluster distribution
    kmeans_model = KMeans(n_clusters=num_clusters, init="k-means++", random_state=0).fit(pretrained_embeddings)
    pre_cluster_labels = kmeans_model.labels_
    deep_k_means = GraphKMeans(60, 60, num_clusters, beta, kmeans_model.cluster_centers_).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(deep_k_means.parameters()), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print("TRAINING")
    embeddings = []

    for epoch in range(epochs):

        loss = 0
        cur_loss = 0
        optimizer.zero_grad()
        deep_k_means.train()

        embeddings = []

        for data, adj_matrix in zip(train_loader, adj_matrices):

            data = data.to(device)
            ori_adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
            
            if WASS_LOSS:

                ori_mst_index, ori_nonmst_index = _compute_birth_death_sets(adj_matrix, numSampledCCs, numSampledCycles)
                ori_mst = torch.take(ori_adj.to(device), ori_mst_index.to(device))
                ori_nonmst = torch.take(ori_adj.to(device), ori_nonmst_index.to(device))

                adj_norm = preprocess_graph(adj_matrix)
                z_pooled, mu_pooled, logvar_pooled = encoder(data.x, adj_norm, data.batch)
            
                embeddings.append(z_pooled)

                decoded = decoder(z_pooled)

                # Reconstruct the adjacency matrix from the top triangle
                recons_adj = torch.zeros((numSampledCCs+1, numSampledCCs+1), dtype=torch.float32).to(device)
                upper_indices = torch.triu_indices(numSampledCCs+1, numSampledCCs+1, offset=1).to(device)
                recons_adj[upper_indices[0], upper_indices[1]] = decoded
                recons_adj = recons_adj + recons_adj.t()

                recons_adj_n = recons_adj.detach().cpu().numpy()
                new_mst_index, new_nonmst_index = _compute_birth_death_sets(recons_adj_n, numSampledCCs, numSampledCycles)

                new_mst = torch.take(recons_adj.to(device), new_mst_index.to(device))
                new_nonmst = torch.take(recons_adj.to(device), new_nonmst_index.to(device))

                # Compute MSE losses
                loss_mst = criterion(new_mst, ori_mst)
                loss_nonmst = criterion(new_nonmst, ori_nonmst)

                n_nodes = data.num_nodes
                loss_kl = kl_loss(mu_pooled, logvar_pooled, n_nodes)

                loss += alpha * loss_mst + alpha * loss_nonmst + gamma * loss_kl

            else:
                
                adj_norm = preprocess_graph(adj_matrix)
                z_pooled, mu_pooled, logvar_pooled = encoder(data.x, adj_norm, data.batch)
            
                embeddings.append(z_pooled)

                z_pooled = decoder(z_pooled)

                # Reconstruct the adjacency matrix from the top triangle
                max_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float32).to(device)
                upper_indices = torch.triu_indices(max_nodes, max_nodes, offset=1).to(device)
                # max_matrix[upper_indices[0], upper_indices[1]] = z_pooled
                max_matrix[upper_indices[0], upper_indices[1]] = z_pooled.squeeze()
                max_matrix = max_matrix + max_matrix.t()

                n_nodes = data.num_nodes
                reshape_adj = max_matrix.to(device).view(-1)[:n_nodes**2]
                reshape_adj = reshape_adj.view(n_nodes, n_nodes)

                loss_mse = criterion(reshape_adj, ori_adj)
                loss_kl = kl_loss(mu_pooled, logvar_pooled, n_nodes)

                loss += alpha * loss_mse + gamma * loss_kl

        loss_k_means = deep_k_means(torch.stack(embeddings), 0.5)
        loss += loss_k_means
        print(loss_k_means.item())
        loss.backward()  # Backpropagate errors immediately
        cur_loss = loss.item()

        optimizer.step()  # Update parameters(Apply gradient updates) once for the entire batch

        # wandb.log({"k_means_loss": loss_k_means.item()})
        # wandb.log({"train_loss": cur_loss})
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {cur_loss}")
    
    # Stack tensors vertically
    embeddings = torch.cat([e.detach() for e in embeddings], dim=0)
    kmeans_model = KMeans(n_clusters=num_clusters, init="k-means++", random_state=0).fit(embeddings.cpu().numpy())
    cluster_labels = kmeans_model.labels_

    return pre_cluster_labels, cluster_labels

def kl_loss(mu, logstd, n_nodes):
    return -0.5 / n_nodes * torch.mean(
        torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))

# def preprocess_graph(adj):
#     adj = sp.coo_matrix(adj)
#     adj_ = adj + sp.eye(adj.shape[0])
#     rowsum = np.array(adj_.sum(1))
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     # return sparse_to_tuple(adj_normalized)
#     tensor = sparse_mx_to_torch_sparse_tensor(adj_normalized)
#     return tensor.to(device)

# modified
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    median_weight = np.median(adj.data)
    adj_ = adj + sp.eye(adj.shape[0]) * median_weight
    rowsum = np.array(adj_.sum(1))
    average_row = np.mean(rowsum)
    row_normalized = rowsum / average_row
    degree_mat_inv_sqrt = sp.diags(np.power(row_normalized, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    tensor = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    return tensor.to(device)

# def preprocess_graph(adj):
#     adj = sp.coo_matrix(adj)
    
#     tensor = sparse_mx_to_torch_sparse_tensor(adj)
#     return tensor.to(device)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj2pers(adj):
    n = len(adj)
    # Assume networks are connected
    G = nx.from_numpy_array(adj)
    T = nx.maximum_spanning_tree(G)
    MSTedges = T.edges(data=True) # compute maximum spanning tree (MST)

    # Convert 2D edge indices to 1D and sort by weight
    MSTindices = [i * n + j for i, j, _ in sorted(MSTedges, key=lambda x: x[2]['weight'])]

    G.remove_edges_from(MSTedges) # remove MST from the original graph
    nonMSTedges = G.edges(data=True) # find the remaining edges (nonMST) as cycles

    nonMSTindices = [i * n + j for i, j, _ in sorted(nonMSTedges, key=lambda x: x[2]['weight'])]

    total_edges = n * (n - 1) // 2
    all_possible_indices = set(range(total_edges))
    non_zero_weight_indices = set(MSTindices + nonMSTindices)
    zero_weight_indices = list(all_possible_indices - non_zero_weight_indices)

    # Include zero-weight indices in the nonMSTindices list
    nonMSTindices = zero_weight_indices + nonMSTindices

    return MSTindices, nonMSTindices

def _compute_birth_death_sets(adj, numSampledCCs, numSampledCycles):
    ccs, cycles = adj2pers(adj)

    # sorted births of ccs as a feature vector
    numCCs = len(ccs)
    ccVec = [math.ceil((i+1)*numCCs/numSampledCCs)-1 for i in range(numSampledCCs)]

    # sorted deaths of cycles as a feature vector
    numCycless = len(cycles)
    cycleVec = [math.ceil((i+1)*numCycless/numSampledCycles)-1 for i in range(numSampledCycles)]

    npCCs = np.array(ccs)
    npCycles = np.array(cycles)
    return torch.from_numpy(npCCs[ccVec]), torch.from_numpy(npCycles[cycleVec])

# Perform 2-hop graph convolution
def convolve_features(X, A):

    # Calculate 2-hop adjacency matrix
    A_2hop = np.dot(A, A).squeeze()

    # Add self-loops to retain current information
    A_with_self_loops = A_2hop + np.eye(A.shape[0])

    # Normalize adjacency matrix by row sums
    row_sums = A_with_self_loops.sum(axis=1).squeeze()
    D_inv = np.diag(1 / row_sums)
    A_normalized = np.dot(D_inv, A_with_self_loops)

    # Perform message passing to update node features
    X_updated = np.dot(A_normalized, X)
    return X_updated

def load_mutag_data():
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    adjacency_matrices = []
    labels = []
    
    for data in dataset:
        # Convert to dense adjacency matrix
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        
        # Compute the degree of each node
        # degrees = adj.sum(dim=1).unsqueeze(1)  # Sum over columns to get the degree
        deg = degree(data.edge_index[0], dtype=torch.float, num_nodes=data.num_nodes).view(-1, 1)

        # Gather degrees of one-hop neighbors for each node
        neighbors_deg = [deg[data.edge_index[0][data.edge_index[1] == i]].view(-1) for i in range(data.num_nodes)]

        # Calculate min, max, mean, std of degrees of neighbors
        min_deg = torch.stack([d.min() if len(d) > 0 else torch.tensor(0.) for d in neighbors_deg]).view(-1, 1)
        max_deg = torch.stack([d.max() if len(d) > 0 else torch.tensor(0.) for d in neighbors_deg]).view(-1, 1)
        mean_deg = torch.stack([d.mean() if len(d) > 0 else torch.tensor(0.) for d in neighbors_deg]).view(-1, 1)
        # Ensures there are at least two data points for std dev calculation
        std_deg = torch.stack([d.std(unbiased=False) if len(d) > 1 else torch.tensor(0.) for d in neighbors_deg]).view(-1, 1)
       

        # Concatenate degree profile features with original node features
        degree_features = torch.cat([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

        # Compute the average weight of connected edges for each node
        # Assuming initially all weights are 1 (as in a binary adjacency matrix), the sum of weights is the degree
        # We can avoid division by zero by setting the average to zero where degree is zero
        avg_weights = torch.where(deg > 0, deg / deg, torch.zeros_like(deg))

        # Compute Eigen Features
        eigvals, eigvecs = eigh(adj.cpu().numpy())
        eigvecs_features = torch.tensor(eigvecs, dtype=torch.float32)

        # node_features = torch.cat([node_features, degrees, avg_weights], dim=1)
        convolved = torch.tensor(convolve_features(data.x, adj))
        node_features = torch.cat([data.x, deg, convolved, degree_features], dim=1)

        # dot product between nodes
        dot_product_matrix = torch.mm(node_features, node_features.t())

        # # ensures only connected nodes have their dot products as weights
        weighted_adj = adj * dot_product_matrix
        # weighted_adj = dot_product_matrix
        # print(weighted_adj)
        # weighted_adj = dot_product_matrix + eigvecs_features

        adjacency_matrices.append(weighted_adj.cpu().numpy())
        labels.append(data.y.item())

    return dataset, adjacency_matrices, labels

def one_tune_instance():

    wandb.init(project="hyperparameter-sweep-top-clustering")
    
    epochs = wandb.config.epochs
    pretrain_epochs = wandb.config.pretrain_epochs
    learning_rate = wandb.config.lr
    numSampledCCs = wandb.config.numSampledCCs
    alpha = wandb.config.alpha
    # alpha = 100
    beta = wandb.config.beta
    # beta = 1
    gamma = wandb.config.gamma
    # gamma = 1
    # latent_space_VAE = wandb.config.latent_space_VAE
    latent_space_VAE = 4

    hyperparameters = (epochs, pretrain_epochs, learning_rate, numSampledCCs, alpha, beta, gamma, latent_space_VAE)

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_loader, adj_matrices, labels_true = load_mutag_data()

    # Pretraining
    pretrain_labels_pred, train_labels_pred = train((train_loader, adj_matrices), hyperparameters)

    pretrain_ari = adjusted_rand_score(labels_true, pretrain_labels_pred)
    wandb.log({"pretrain_ari": pretrain_ari})

    train_ari = adjusted_rand_score(labels_true, train_labels_pred)
    wandb.log({"train_ari": train_ari})

def main():

    TUNE_HYPERPARAMETERS = False

    if TUNE_HYPERPARAMETERS:
        sweep_config = {
            'method': 'random',  # or 'grid' or 'bayes'
            'metric': {
                'name': 'train_ari',
                'goal': 'maximize'
            },
            'parameters': {
                'lr': {
                    'max': 1e-1, 'min': 1e-4, 'distribution': 'log_uniform_values'
                },
                'epochs': {
                    'values': list(range(50, 200, 10))
                },
                'pretrain_epochs': {
                    'values': list(range(30, 100, 10))
                },
                'numSampledCCs': {
                    'values': list(range(4, 20))
                },
                'alpha': {
                    'values': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
                },
                'beta': {
                    'values': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
                },
                'gamma': {
                    'values': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
                },
                # 'latent_space_VAE': {
                #     'values': [3, 4, 5, 6]
                # },
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_config, project='hyperparameter-sweep')
        wandb.agent(sweep_id, function=one_tune_instance, count=60)

    else:

        print("NOT TUNING PARAMS")

        # Manually set here per run:
        epochs, pretrain_epochs = 30, 30
        learning_rate = 0.01
        numSampledCCs = 12

        alpha, beta, gamma = 0.5, 1000, 1000
        latent_space_VAE = 5

        hyperparameters = (epochs, pretrain_epochs, learning_rate, numSampledCCs, alpha, beta, gamma, latent_space_VAE)

        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        train_loader, adj_matrices, labels_true = load_mutag_data()

        # Pretraining
        pretrain_labels_pred, train_labels_pred = train((train_loader, adj_matrices), hyperparameters)
        
        # Get ari prints
        pretrain_ari = adjusted_rand_score(labels_true, pretrain_labels_pred)
        print(f"Adjusted Rand Index after Pretraining: {pretrain_ari}")  

        train_ari = adjusted_rand_score(labels_true, train_labels_pred)
        print(f"Adjusted Rand Index after Fine-tuning: {train_ari}")  

if __name__ == '__main__':
    main()
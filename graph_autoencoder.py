import sys
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import ray
import wandb

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.cluster import contingency_matrix
from matplotlib import pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv_1 = GCNConv(7, 64)
        self.activ_1 = nn.ReLU()
        self.conv_2 = GCNConv(64, 32)
        self.activ_2 = nn.ReLU()
        self.linear = nn.Linear(32, 5)

    def forward(self, x, edge_index, batch):
        x = self.conv_1(x, edge_index)
        x = self.activ_1(x)
        x = self.conv_2(x, edge_index)
        x = self.activ_2(x)
        x = self.linear(x)

        x = global_mean_pool(x, batch)  # batch is the index of the batch to which the nodes belong
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
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

def pretrain(dataset, new_adj_dim, numSampledCCs, numSampledCycles, epochs=80, lr=1e-4):

    print("PRETRAINING")

    train_loader, adj_matrices, labels_true = dataset

    encoder = GNN().to(device)
    decoder = MLP(5, new_adj_dim**2).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()

    final_embeddings = []

    for epoch in range(epochs):

        encoder.train()
        decoder.train()
        total_loss = 0

        for data, adj_matrix in zip(train_loader, adj_matrices):

            data = data.to(device)

            x = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
            ori_mst_index, ori_nonmst_index = _compute_birth_death_sets(adj_matrix, numSampledCCs, numSampledCycles)
            ori_mst = torch.take(x, ori_mst_index).to(device)
            ori_nonmst = torch.take(x, ori_nonmst_index).to(device)

            encoded = encoder(data.x, data.edge_index, data.batch)
            
            if epoch == epochs-1:
                final_embeddings.append(torch.squeeze(encoded).detach().numpy())

            decoded = decoder(encoded)
            decoded = decoded.view(new_adj_dim, new_adj_dim)
            
            # new_mst_index = convert_index(ori_mst_index, data.num_nodes, new_adj_dim)
            # new_nonmst_index = convert_index(ori_nonmst_index, data.num_nodes, new_adj_dim)

            decoded_adj = decoded.detach().numpy()
            new_mst_index, new_nonmst_index = _compute_birth_death_sets(decoded_adj, numSampledCCs, numSampledCycles)

            new_mst = torch.take(decoded, new_mst_index).to(device)
            new_nonmst = torch.take(decoded, new_nonmst_index).to(device)

            # Compute MSE losses
            loss_mst = criterion(new_mst, ori_mst)
            loss_nonmst = criterion(new_nonmst, ori_nonmst)

            # Sum the losses
            loss = loss_mst + loss_nonmst
            loss.backward()  # Backpropagate errors immediately
            total_loss += loss.item()

        optimizer.step()  # Update parameters(Apply gradient updates) once for the entire batch
        optimizer.zero_grad() # Reset gradients after update

        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Average Loss: {total_loss / len(dataset)}")

    return encoder, decoder, final_embeddings

def train(dataset, hyperparameters, num_clusters=2):

    # train_loader, adj_matrices = dataset
    train_loader, adj_matrices, labels_true = dataset
    epochs, pretrain_epochs, new_adj_dim, learning_rate, numSampledCCs, numSampledCycles = hyperparameters

    encoder, decoder, pretrained_embeddings = pretrain(dataset, new_adj_dim, numSampledCCs=numSampledCCs, numSampledCycles=numSampledCycles, epochs=pretrain_epochs, lr=learning_rate)

    # Run k-means++ to get initial cluster distribution
    kmeans_model = KMeans(n_clusters=num_clusters, init="k-means++").fit(pretrained_embeddings)
    pre_cluster_labels = kmeans_model.labels_
    deep_k_means = GraphKMeans(60, 60, num_clusters, 0.1, kmeans_model.cluster_centers_).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(deep_k_means.parameters()), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print("TRAINING")
    embeddings = []

    for epoch in range(epochs):
        loss = 0

        total_loss = 0
        deep_k_means.train()

        embeddings = []  # List to store embeddings

        for data, adj_matrix in zip(train_loader, adj_matrices):

            data = data.to(device)

            x = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
            ori_mst_index, ori_nonmst_index = _compute_birth_death_sets(adj_matrix, numSampledCCs, numSampledCycles)
            ori_mst = torch.take(x, ori_mst_index).to(device)
            ori_nonmst = torch.take(x, ori_nonmst_index).to(device)

            encoded = encoder(data.x, data.edge_index, data.batch)
            embeddings.append(encoded)
            
            decoded = decoder(encoded)
            decoded = decoded.view(new_adj_dim, new_adj_dim)

            # new_mst_index = convert_index(ori_mst_index, data.num_nodes, new_adj_dim)
            # new_nonmst_index = convert_index(ori_nonmst_index, data.num_nodes, new_adj_dim)

            decoded_adj = decoded.detach().numpy()
            new_mst_index, new_nonmst_index = _compute_birth_death_sets(decoded_adj, numSampledCCs, numSampledCycles)

            new_mst = torch.take(decoded, new_mst_index).to(device)
            new_nonmst = torch.take(decoded, new_nonmst_index).to(device)

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

        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Average Loss: {total_loss / len(dataset)}")
    
    # Stack tensors vertically
    embeddings = torch.cat([e.detach() for e in embeddings], dim=0)
    kmeans_model = KMeans(n_clusters=num_clusters, init="k-means++").fit(embeddings.numpy())
    cluster_labels = kmeans_model.labels_

    train_ari = adjusted_rand_score(labels_true, cluster_labels)
    wandb.log({"train_ari": train_ari})

    # return pre_cluster_labels, cluster_labels
    return {"ari": train_ari}


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

def adj2pers(adj):
    n = len(adj)
    # Assume networks are connected
    G = nx.from_numpy_array(adj)
    T = nx.maximum_spanning_tree(G)
    MSTedges = T.edges(data=True) # compute maximum spanning tree (MST)
    # print(MSTedges)

    # Convert 2D edge indices to 1D and sort by weight
    MSTindices = [i * n + j for i, j, _ in sorted(MSTedges, key=lambda x: G[x[0]][x[1]]['weight'])]

    # ccs = sorted([cc[2]['weight'] for cc in MSTedges], reverse=True) # sort all weights in MST
    G.remove_edges_from(MSTedges) # remove MST from the original graph
    nonMSTedges = G.edges(data=True) # find the remaining edges (nonMST) as cycles

    nonMSTindices = [i * n + j for i, j, _ in sorted(nonMSTedges, key=lambda x: G[x[0]][x[1]]['weight'])]

    # cycles = sorted([cycle[2]['weight'] for cycle in nonMSTedges], reverse=True) # sort all weights in nonMST
    numTotalEdges = (len(adj) * (len(adj) - 1)) / 2
    numZeroEdges = numTotalEdges - len(MSTindices) - len(nonMSTindices)
    if numZeroEdges != 0:
        nonMSTindices.extend(int(numZeroEdges) * [0]) # extend 0-valued edges
    return MSTindices, nonMSTindices

def _compute_birth_death_sets(adj, numSampledCCs=9, numSampledCycles=36):
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

def convert_index(indices, old_dim, new_dim):
    new_total_elements = new_dim ** 2

    # Map each index in the larger dimension to the smaller dimension
    # Using modulo to ensure all indices fit within the new dimension
    new_indices = indices % new_total_elements
    return new_indices

def load_mutag_data():
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    adjacency_matrices = []
    labels = []
    
    for data in dataset:
        # Convert to dense adjacency matrix
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        
        # Retrieve node features and compute the dot product matrix
        node_features = data.x
        dot_product_matrix = torch.mm(node_features, node_features.t())

        # # ensures only connected nodes have their dot products as weights
        # weighted_adj = adj * dot_product_matrix

        adjacency_matrices.append(dot_product_matrix.numpy())
        labels.append(data.y.item())

    return dataset, adjacency_matrices, labels

def one_tune_instance():

    wandb.init(project="hyperparameter-sweep")
    
    epochs = wandb.config.epochs
    pretrain_epochs = wandb.config.pretrain_epochs
    new_adj_dim = wandb.config.new_adj_dim
    learning_rate = wandb.config.lr
    numSampledCCs = wandb.config.numSampledCCs
    numSampledCycles = wandb.config.numSampledCycles

    hyperparameters = (epochs, pretrain_epochs, new_adj_dim, learning_rate, numSampledCCs, numSampledCycles)

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    train_loader, adj_matrices, labels_true = load_mutag_data()

    # Pretraining
    pretrain_labels_pred, train_labels_pred = train((train_loader, adj_matrices, labels_true), hyperparameters)
    pretrain_ari = adjusted_rand_score(labels_true, pretrain_labels_pred)

    wandb.log({"pretrain_ari": pretrain_ari})
    print(f"Adjusted Rand Index after Pretraining: {pretrain_ari}")

    # # Fine-tuning
    # train_ari = adjusted_rand_score(labels_true, train_labels_pred)
    # print(f"Adjusted Rand Index after Fine-tuning: {train_ari}")  

    # wandb.log({"train_ari": train_ari})


def main(num_samples=50, max_num_epochs=200, gpus_per_trial=1):
    # np.random.seed(0)
    # random.seed(0)
    # torch.manual_seed(0)

    sweep_config = {
        'method': 'random',  # or 'grid' or 'bayes'
        'metric': {
            'name': 'train_ari',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'max': 1e-1, 'min': 1e-5, 'distribution': 'log_uniform_values'
            },
            'epochs': {
                'values': list(range(50, 200, 10))
            },
            'new_adj_dim': {
                'values': list(range(5, 15))
            },
            'pretrain_epochs': {
                'values': list(range(30, 100, 10))
            },
            'numSampledCCs': {
                'values': list(range(4, 18))
            },
            'numSampledCycles': {
                'values': list(range(6, 60))
            }
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project='hyperparameter-sweep')
    wandb.agent(sweep_id, function=one_tune_instance, count=100)

    # Load the MUTAG dataset
    # train_loader, adj_matrices, labels_true = load_mutag_data()

    # data = (train_loader, adj_matrices, labels_true)

    # # Pretraining
    # pretrain_labels_pred, train_labels_pred = train((train_loader, adj_matrices), new_adj_dim = 10)
    # pretrain_ari = adjusted_rand_score(labels_true, pretrain_labels_pred)
    # print(f"Adjusted Rand Index after Pretraining: {pretrain_ari}")

    # # Fine-tuning
    # train_ari = adjusted_rand_score(labels_true, train_labels_pred)
    # print(f"Adjusted Rand Index after Fine-tuning: {train_ari}")  

if __name__ == '__main__':
    main()
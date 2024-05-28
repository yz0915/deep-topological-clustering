import tensorflow as tf


def fc_layers(input, specs):
    """
    Parameters:
    - input: graph's node features (a feature matrix where each row represents a node and each column represents a feature)

    Returns:
    -input: transformed input after it has passed through all the layers

    """
    [dimensions, activations, names] = specs
    for dimension, activation, name in zip(dimensions, activations, names):
        # create dense layers
        input = tf.compat.v1.layers.dense(
            inputs=input,
            units=dimension,
            activation=activation,
            name=name,
            reuse=tf.compat.v1.AUTO_REUSE,
        )
    return input


def graph_autoencoder(input_features, specs):
    """
    Implements an autoencoder for graph data.

    Parameters:
    - input_features: Tensor, the feature matrix of the graph where each row corresponds to a node.
    - specs: A list containing three lists: dimensions, activations, and names for layers.

    Returns:
    - embedding: The embedded representation of the graph (output of the encoder).
    - output: The reconstructed feature matrix of the graph (output of the decoder).
    """

    [dimensions, activations, names] = specs
    mid_ind = int(len(dimensions) / 2)

    # Encoder
    embedding = fc_layers(
        input_features, [dimensions[:mid_ind], activations[:mid_ind], names[:mid_ind]]
    )

    # Decoder
    output = fc_layers(
        embedding, [dimensions[mid_ind:], activations[mid_ind:], names[mid_ind:]]
    )

    return embedding, output


num_nodes = 100  # 100 nodes in the graph
feature_dim = 20  # Each node is represented by 20 features
input_features = tf.random.normal([num_nodes, feature_dim])

specs = [
    [512, 256, 128, 256, 512],
    [tf.nn.relu] * 5,
    ["enc1", "enc2", "embed", "dec1", "dec2"],
]

embedding, reconstructed = graph_autoencoder(input_features, specs)

# Print the embedding and reconstructed tensors
print("Embedding:\n", embedding.numpy())
print("Reconstructed:\n", reconstructed.numpy())

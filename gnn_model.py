import numpy as np
import networkx as nx
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from itertools import combinations

# Function to initialize weights and biases for each layer
def initialize_parameters(node_dim, edge_dim, hidden_dim, output_dim):
    parameters = {}
    parameters['W1'] = np.random.randn(node_dim, hidden_dim)
    parameters['b1'] = np.zeros((1, hidden_dim))
    parameters['W2'] = np.random.randn(edge_dim, hidden_dim)
    parameters['b2'] = np.zeros((1, hidden_dim))
    parameters['W3'] = np.random.randn(hidden_dim, output_dim)
    parameters['b3'] = np.zeros((1, output_dim))
    return parameters

# Activation function (e.g., ReLU)
def relu(x):
    return np.maximum(0, x)

# Message passing function
def message_passing(nodes, edges, parameters):
    # nodes_combinations = list(combinations(nodes, 2))
    # print('combinations of nodes:')
    # for combo in nodes_combinations:
    #     print(combo)
    # print('list of edge features')
    # print(edges)
    # h1_nodes = relu(np.dot(nodes, parameters['W1']) + parameters['b1'])
    h1_edges = relu(np.dot(edges, parameters['W2']) + parameters['b2'])
    
    # Combine node and edge features
    # print(h1_nodes.shape)
    # print(h1_edges.shape)
    # combined_features = np.concatenate([h1_nodes, h1_edges], axis=0)
    # print(combined_features.shape)
    print(parameters['W3'].shape)
    h2 = np.dot(h1_edges, parameters['W3']) + parameters['b3']
    return h2

# Training loop
def train_mpnn(train_graphs, test_graphs, learning_rate, epochs):
    node_dim = 3 # Dimension of node features
    edge_dim = 2  # Dimension of edge features
    hidden_dim = 64  # You can adjust this based on your requirements
    output_dim = 1  # Predicting a single scalar coupling constant

    parameters = initialize_parameters(node_dim, edge_dim, hidden_dim, output_dim)

    for epoch in range(epochs):
        total_loss = 0.0
        for graph in train_graphs:
            nodes = np.array(list(graph.nodes(data=True)))
            node_features = np.array([list(node[1].values()) for node in nodes])
            # Extract only the 3rd, 4th, and 5th features from node data
            nodes = node_features[:, 2:5].astype(float)

            # nodes = nodes.astype(float)
            # print(nodes)
            edges = np.array(list(graph.edges(data=True)))
            edge_features = np.array([list(edge[2].values()) for edge in edges])
            edges = edge_features
            edges = edge_features[:, 0:2].astype(float)

            # print(edges)

            # Assuming 'SCC' is the key for scalar coupling constant in the edge data
            labels = np.array([edge[1] for edge in edges])
            print('labels:')
            print(labels)
            # Forward pass
            predictions = message_passing(nodes, edges, parameters)
            predictions = np.ravel(predictions)
            print('predictions:')
            print(predictions)

            # Compute loss (mean squared error)
            loss = np.mean((predictions - labels) ** 2)
            total_loss += loss

            # Backward pass (gradient descent)
            d_loss = 2 * (predictions - labels) / len(labels)
            print(predictions - labels)
            print(d_loss)
            # Update parameters
            gradient = np.dot(message_passing(nodes, edges, parameters).T, d_loss)
            print(gradient.shape)
            parameters['W3'] -= learning_rate * gradient.T  # Transpose gradient
            parameters['b3'] -= learning_rate * np.sum(d_loss, axis=0, keepdims=True)

            # Make sure the shapes are aligned
            assert gradient.shape == parameters['W3'].shape



        # Print average loss for monitoring training progress
        if epoch % 100 == 0:
            average_loss = total_loss / len(train_graphs)
            print(f'Epoch {epoch}, Train Average Loss: {average_loss}')

    # Make predictions on the test set
    test_predictions = []
    test_labels = []
    for graph in test_graphs:
        nodes = np.array(list(graph.nodes(data=True)))
        edges = np.array(list(graph.edges(data=True)))

        # Assuming 'SCC' is the key for scalar coupling constant in the edge data
        labels = np.array([edge[2]['SCC'] for edge in edges])

        # Forward pass
        predictions = message_passing(nodes, edges, parameters)
        
        test_predictions.extend(predictions)
        test_labels.extend(labels)

    # Calculate Mean Absolute Error (MAE) on the test set
    test_mae = mean_absolute_error(test_labels, test_predictions)
    print(f'Test MAE: {test_mae}')

# Load graphs from pickle files for training and testing
with open('graphs_train.pkl', 'rb') as file:
    loaded_graphs = pickle.load(file)
    
train_graphs, test_graphs = train_test_split(loaded_graphs, test_size=0.2, random_state=42)
# Train the MPNN and evaluate on the test set
learning_rate = 0.001
epochs = 1000
train_mpnn(train_graphs, test_graphs, learning_rate, epochs)

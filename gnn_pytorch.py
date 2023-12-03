import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Define the MPNN model
class MPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(MPNN, self).__init__()
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.fc2 = nn.Linear(edge_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, nodes, edges):
        h1_nodes = torch.relu(self.fc1(nodes))
        h1_edges = torch.relu(self.fc2(edges))
        combined_features = torch.cat([h1_nodes, h1_edges], dim=0)
        h2 = self.fc3(h1_edges)
        return h2

# Training loop
def train_mpnn(train_graphs, test_graphs, learning_rate, epochs):
    node_dim = 3  # Dimension of node features
    edge_dim = 2  # Dimension of edge features
    hidden_dim = 64  # You can adjust this based on your requirements
    output_dim = 1  # Predicting a single scalar coupling constant

    model = MPNN(node_dim, edge_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for graph in train_graphs:
            nodes = torch.tensor([list(node[1].values())[2:5] for node in graph.nodes(data=True)], dtype=torch.float32)
            edges = torch.tensor([list(edge[2].values())[:2] for edge in graph.edges(data=True)], dtype=torch.float32)
            labels = torch.tensor([edge[1] for edge in graph.edges(data=True)], dtype=torch.float32)

            optimizer.zero_grad()
            predictions = model(nodes, edges).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for monitoring training progress
        if epoch % 100 == 0:
            average_loss = total_loss / len(train_graphs)
            print(f'Epoch {epoch}, Train Average Loss: {average_loss}')

    # Make predictions on the test set
    test_predictions = []
    test_labels = []
    for graph in test_graphs:
        nodes = torch.tensor([list(node[1].values())[2:5] for node in graph.nodes(data=True)], dtype=torch.float32)
        edges = torch.tensor([list(edge[2].values())[:2] for edge in graph.edges(data=True)], dtype=torch.float32)
        labels = torch.tensor([edge[2]['SCC'] for edge in graph.edges(data=True)], dtype=torch.float32)

        predictions = model(nodes, edges).squeeze()
        test_predictions.extend(predictions.detach().numpy())
        test_labels.extend(labels.numpy())

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

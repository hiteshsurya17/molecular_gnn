import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

with open('graphs_train.pkl', 'rb') as file:
    loaded_graphs = pickle.load(file)
print(len(loaded_graphs))
# Initialize an empty list to store DataFrame objects
dfs = []

for graph in loaded_graphs:
    edges = np.array(list(graph.edges(data=True)))
    edge_features = np.array([list(edge[2].values()) for edge in edges])
    edges = edge_features
    # print(edges)
    new_data = {'distance': edges[:, 0], 'SSC': edges[:, 1] , 'Ctype': edges[:, 2]}
    df = pd.DataFrame(new_data)
    
    dfs.append(df)

edges_data = pd.concat(dfs, ignore_index=True)
ctype_encoded = pd.get_dummies(edges_data['Ctype'], prefix='Ctype')
ctype_encoded = ctype_encoded.astype(int)
edges_data = pd.concat([edges_data, ctype_encoded], axis=1)
edges_data = edges_data.drop('Ctype', axis=1)

# print(edges_data.head(5))

# Extract features (all columns except 'SSC')
edges_data['SSC'] = pd.to_numeric(edges_data['SSC'], errors='coerce')

# Drop rows with NaN values in the 'SSC' column
edges_data = edges_data.dropna(subset=['SSC'])
X = torch.tensor(edges_data.drop('SSC', axis=1).astype(float).values, dtype=torch.float32)

# Extract the target variable 'SSC'
y = torch.tensor(edges_data['SSC'].values, dtype=torch.float32).view(-1, 1)


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
class SimpleANN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleANN(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training loss for monitoring
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Training Loss: {loss.item()}')

# Make predictions on the test set
with torch.no_grad():
    test_predictions = model(X_test)

# Calculate Mean Absolute Error (MAE) on the test set
test_mae = mean_absolute_error(y_test, test_predictions.numpy())
print(f'Test MAE: {test_mae}')

y_test_np = y_test.numpy()
test_predictions_np = test_predictions.numpy()

# Calculate R^2 score
r2 = r2_score(y_test_np, test_predictions_np)
print(f'R^2 Score: {r2}')

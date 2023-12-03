import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Load the list of graphs from the current working directory
with open('graphs_train.pkl', 'rb') as file:
    loaded_graphs = pickle.load(file)

print('nuber of graphs:',len(loaded_graphs))
example_graph = loaded_graphs[0]

# Visualizing the graph
pos = nx.spring_layout(example_graph) 
plt.show()
nx.draw(example_graph, pos, with_labels=False, font_weight='bold', node_size=700, node_color='lightblue')

# Drawing additional node data as labels
node_labels = {node: f"{example_graph.nodes[node]['atom_type']}"
               for node in example_graph.nodes()}
nx.draw_networkx_labels(example_graph, pos, labels=node_labels, font_color='black', font_size=8)

# drawing edge distance information
edge_labels = {(u, v): f"D={example_graph.edges[u, v]['distance']:.2f},J={example_graph.edges[u, v]['SCC']:.2f},T={example_graph.edges[u, v]['Ctype']}" for u, v in example_graph.edges()}
nx.draw_networkx_edge_labels(example_graph, pos, edge_labels=edge_labels, font_color='red')

# Show the plot
plt.show()
# Print node and edge data (for demonstration purposes)
print("Nodes of example graph:")
for node, data in example_graph.nodes(data=True):
    print(f"Node {node}: {data}")

    
print("\nEdges of example graph:")
for edge in example_graph.edges(data=True):
    node1, node2, data = edge
    print(f"Edge ({node1}, {node2}): {data}")

print('edge data:')
print(example_graph.edges(data=True))
print("\labels of example graph:")
print(f"labels:{example_graph['labels']}")



adjacent_mat_list = []
for graph in loaded_graphs:
    adjacency_matrix = nx.adjacency_matrix(graph)
    adjacent_mat_list.append(adjacency_matrix)
print('number of adjacency matrices:',len(adjacent_mat_list))

print('first garph adjacency matrix: ',adjacent_mat_list[0])


# data = []
# labels = []
# for g in loaded_graphs:
#     for n,d in g.nodes(data=True):
#         print(d)
#         # l = d['']
#         break
#     break

# train_data, test_data, train_labels, test_labels = train_test_split(data , labels, test_size=0.20)

# print('number of training graphs and lables:',len(train_data),len(train_labels))
# print('number of testing graphs and lables:',len(test_data),len(test_labels))
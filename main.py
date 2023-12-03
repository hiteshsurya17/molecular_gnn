import pandas as pd 
import networkx as nx
import pickle
import numpy as np

train_data = pd.read_csv(r'/Users/hiteshchowdarysuryadevara/Downloads/champs-scalar-coupling/train.csv')
test_data = pd.read_csv(r'/Users/hiteshchowdarysuryadevara/Downloads/champs-scalar-coupling/test.csv')
structures = pd.read_csv(r'/Users/hiteshchowdarysuryadevara/Downloads/champs-scalar-coupling/structures.csv')
# dipole_moments = pd.read_csv(r'/Users/hiteshchowdarysuryadevara/Downloads/champs-scalar-coupling/dipole_moments.csv')
# magnetic_tensors = pd.read_csv(r'/Users/hiteshchowdarysuryadevara/Downloads/champs-scalar-coupling/magnetic_shielding_tensors.csv')
# mulliken_charges = pd.read_csv(r'/Users/hiteshchowdarysuryadevara/Downloads/champs-scalar-coupling/mulliken_charges.csv')
# potential_energy = pd.read_csv(r'/Users/hiteshchowdarysuryadevara/Downloads/champs-scalar-coupling/potential_energy.csv')
# scalar_coupling_contributions = pd.read_csv(r'/Users/hiteshchowdarysuryadevara/Downloads/champs-scalar-coupling/scalar_coupling_contributions.csv')

print('Train_data:')
print(train_data.head())
print('structures_data:')
print(structures.head())
print("null values in train_data:")
print(train_data.isna().sum())


def create_molecule_graph(molecule_name):
    # Extract structure data for the molecule
    molecule_data = structures[structures['molecule_name'] == molecule_name]
    coupling_constant_data = train_data[train_data['molecule_name'] == molecule_name]

    # Create a graph
    G = nx.Graph()

    # Add nodes (atoms)
    for idx, atom in molecule_data.iterrows(): #idx is the index and atom will be the panadas series of the row
        G.add_node(atom['atom_index'],
                   atom_type=atom['atom'],
                   molecule_name=molecule_name,
                   x=atom['x'],
                   y=atom['y'],
                   z=atom['z'])

    # Add edges (bonds)
    for idx, atom in molecule_data.iterrows():
        for idx2, atom2 in molecule_data.iterrows():
            if idx != idx2:  # Avoid self-loops
                distance = ((atom['x'] - atom2['x'])**2 +
                            (atom['y'] - atom2['y'])**2 +
                            (atom['z'] - atom2['z'])**2) ** 0.5
                

                # Example: Add an edge if the distance is below a threshold
                # if distance < 1.5:
                G.add_edge(atom['atom_index'], atom2['atom_index'], distance=distance)
    
    # adding coupling type and coupling constants to these edges 
    for u, v, data in G.edges(data=True):

        #SCC = scalar_coupling_constant Ctype =coupling type
        result = coupling_constant_data.loc[(coupling_constant_data[['atom_index_0', 'atom_index_1']].isin([u, v])).all(axis=1), ['scalar_coupling_constant', 'type']]
        if not result.empty:
            G[u][v]['SCC'] = result['scalar_coupling_constant'].iloc[0]
            G[u][v]['Ctype'] = result['type'].iloc[0]
        else :
            G[u][v]['SCC'] = np.nan
            G[u][v]['Ctype'] = np.nan
    return G

# creating a list of molecule names in train data to fect graphs structure from structures data of only training molecules
unique_molecule_names = train_data['molecule_name'].unique()
unique_molecule_names_list = list(unique_molecule_names)
unique_molecule_names_list = unique_molecule_names_list[0:1000] #TOTAL 85012
print('lenght of unique molecule names:',len(unique_molecule_names_list))

#graps list to save all the graphs of training data molecules
graphs_train = []
n=0

for molecule_name in unique_molecule_names_list:
    graph = create_molecule_graph(molecule_name)
    graphs_train.append(graph)
    n+=1
    if n%100 == 0:
        print('loading graphs:',round(n/10000*100,2),'%')
    if n == 10000:
        break

print('number of graphs :',len(graphs_train))

# Save the list of graphs in the current working directory
with open('graphs_train.pkl', 'wb') as file:
    pickle.dump(graphs_train, file)

# Print node and edge data (for demonstration purposes)
# print("Nodes:")
# for node, data in graph_example.nodes(data=True):
#     print(f"Node {node}: {data}")

# print("\nEdges:")
# for edge in graph_example.edges(data=True):
#     node1, node2, data = edge
#     print(f"Edge ({node1}, {node2}): {data}")


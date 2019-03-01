import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import torch.utils.data

# """DATASET PREPROCESSING"""
# # Open the csv file with the datasetset
# dataset = pd.read_csv('metabolic_mania_data.csv')

"""OPEN FORMATTED DATASET"""
dataset = pd.read_csv(r"metabolic_mania_data_norm_transpose.csv")

input_shape = len(dataset)

# Define all datasets to be used in training and turn them into lists
connected_components_int = torch.FloatTensor(list(dataset.connected_components_int))
diameter_int = torch.FloatTensor(list(dataset.diameter_int))
radius_int = torch.FloatTensor(list(dataset.radius_int))
shortest_path_int = torch.FloatTensor(list(dataset.shortest_path_int))
characteristic_path_len = torch.FloatTensor(list(dataset.characteristic_path_len))
avg_num_neighbours = torch.FloatTensor(list(dataset.avg_num_neighbours))
density_real = torch.FloatTensor(list(dataset.density_real))
heterogeniety_real = torch.FloatTensor(list(dataset.heterogeniety_real))
isolated_nodes_int = torch.FloatTensor(list(dataset.isolated_nodes_int))
number_self_loops = torch.FloatTensor(list(dataset.number_self_loops))
multi_edge_node_pair_int = torch.FloatTensor(list(dataset.multi_edge_node_pair_int))
neigh_connect = torch.FloatTensor(list(dataset.neigh_connect))
neigh_connect = torch.FloatTensor(list(dataset.neigh_connect))
stress = torch.FloatTensor(list(dataset.stress))
partner_multi_edge_node_int = torch.FloatTensor(list(dataset.partner_multi_edge_node_int))
degree = torch.FloatTensor(list(dataset.degree))
topo_coef = torch.FloatTensor(list(dataset.topo_coef))
between_cent = torch.FloatTensor(list(dataset.between_cent))
radialiy = torch.FloatTensor(list(dataset.radialiy))
eccentrcity = torch.FloatTensor(list(dataset.eccentrcity))
close_centrality = torch.FloatTensor(list(dataset.close_centrality))
avg_short_path_len = torch.FloatTensor(list(dataset.avg_short_path_len))
cluster_coef = torch.FloatTensor(list(dataset.cluster_coef))
node_count = torch.FloatTensor(list(dataset.node_count))
edge_count = torch.FloatTensor(list(dataset.edge_count))

datasets = []
datasets.extend((
                connected_components_int,
                diameter_int,
                radius_int,
                shortest_path_int,
                characteristic_path_len,
                avg_num_neighbours,
                density_real,
                heterogeniety_real,
                isolated_nodes_int,
                number_self_loops,
                multi_edge_node_pair_int,
                neigh_connect,
                neigh_connect,
                stress,
                partner_multi_edge_node_int,
                degree,
                topo_coef,
                between_cent,
                radialiy,
                eccentrcity,
                close_centrality,
                avg_short_path_len,
                cluster_coef,
                node_count,
                edge_count,
                ))

# # Normnalizer for all datasets
# processed_datasets = []

# def normalizer(datasets):
#     for dataset in datasets:
#         new_dataset = [(datapoint - float(min(dataset))) / (float(max(dataset)) - float(min(dataset))) for datapoint in dataset]
#         processed_datasets.append(new_dataset)
#     return processed_datasets

# normalizer(datasets)

# """WRITTING NEW CSV FILE"""
# # Writting normalized dataset into a new file using csv
# def writeCsvFile(fname, data, *args, **kwargs):
#     mycsv = csv.writer(open(fname, 'w'), *args, **kwargs)
#     for row in processed_datasets:
#       mycsv.writerow(row)

# writeCsvFile(r'metabolic_mania_data_norm.csv', processed_datasets)

"""NEURAL NETWORK"""
class Neural_Network(nn.Module):
        def __init__(self):
                super(Neural_Network, self).__init__()
                self.layer_1 = nn.Linear(512, 256)
                self.layer_2 = nn.Linear(256, 128)
                self.layer_3 = nn.Linear(128, 64)
                self.layer_4 = nn.Linear(64, 32)
                self.layer_5 = nn.Linear(32, 16)
                self.layer_6 = nn.Linear(16, 1)

                self.sigmoid = nn.Sigmoid()
                self.relu = nn.ReLU()

        def forward(self, X):
                out_1 = self.relu(self.layer_1(X))
                out_2 = self.sigmoid(self.layer_2(out_1))
                out_3 = self.sigmoid(self.layer_3(out_2))
                out_4 = self.sigmoid(self.layer_4(out_3))
                out_5 = self.sigmoid(self.layer_5(out_4))
                prediction = self.relu(self.layer_6(out_5))
                return prediction

# Creating a variable for our model
model = Neural_Network()

"""COMPILING, FITTING AND RUNNING"""

# Construct loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

# Batching datasets
X_data = datasets[2] # Choose from 25 different X datasets
Y_data = datasets[24] # Choose a different dataset for the Y dataset

X_data = torch.utils.data.DataLoader(X_data, batch_size = 512)
Y_data = torch.utils.data.DataLoader(Y_data, batch_size = 512)

X_data_batch = next(iter(X_data))
Y_data_batch = next(iter(Y_data))

print 

# Construct training loop
epochs = 100

running_loss = 0.0
for epoch in range(epochs):

        # Define Variables
        prediction = model(X_data_batch)
        loss = criterion(prediction, Y_data_batch) # Choose a different dataset for the Y dataset

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if epoch % 10 == 9:
            print('Epochs: %5d | Loss: %.3f' % (epoch + 1, running_loss / 10))
            running_loss = 0.0

# Saving model weights per epoch
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             })
import torch 
import torch.nn.functional as F
from torch.nn import Linear 
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATModel, self).__init__()
        self.gat_conv = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.lin = Linear(hidden_channels * heads, out_channels)
        self.attention_weights = None

    def forward(self, x, edge_index, batch):
        x, self.attention_weights = self.gat_conv(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

    def get_attention_weights(self):
        return self.attention_weights


class MLPModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.17988426707538552):
        super(MLPModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(Linear(prev_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size)) # Add BatchNorm
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))  # Add Dropout
            prev_size = hidden_size
        layers.append(Linear(prev_size, output_size))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

#change the name of CombinedModel to CombinedModelGATMLP
class CombinedModelGATMLP(torch.nn.Module):
    def __init__(self, gnn_in_channels, gnn_hidden_channels, gnn_out_channels,
                 mlp_input_size, mlp_hidden_sizes, mlp_output_size, dense_size_1, dense_size_2):
        super(CombinedModelGATMLP, self).__init__()
        self.gnn = GATModel(gnn_in_channels, gnn_hidden_channels, gnn_out_channels)
        self.mlp = MLPModel(mlp_input_size, mlp_hidden_sizes, mlp_output_size)  
        self.fc1 = Linear(gnn_out_channels + mlp_output_size, dense_size_1)
        self.batchnorm1 = torch.nn.BatchNorm1d(dense_size_1)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.linear = Linear(dense_size_1, 1)

    def forward(self, data, fingerprints):
        gnn_out = self.gnn(data.x, data.edge_index, data.batch)
        mlp_out = self.mlp(fingerprints)
        combined = torch.cat((gnn_out, mlp_out), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.batchnorm1(x)
        x = self.dropout2(x)
        x = self.linear(x)
        return x

    def get_attention_weights(self):
        return self.gnn.get_attention_weights()
import torch
from torch_geometric.data import Batch
class SynchronizedDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data, fingerprint_data):
        assert len(graph_data) == len(fingerprint_data), "Mismatch between graph and fingerprint data"
        self.graph_data = graph_data
        self.fingerprint_data = fingerprint_data
        
    def __len__(self):
        return len(self.graph_data)
        
    def __getitem__(self, idx):
        return self.graph_data[idx], self.fingerprint_data[idx]
        
def collate_fn(batch):
    try:
        graphs = [b[0] for b in batch]
        fingerprints = torch.stack([b[1] for b in batch])
        batch_graphs = Batch.from_data_list(graphs)
        return batch_graphs, fingerprints

    except Exception as e:
        print("Encountered an error: {e}")
        raise


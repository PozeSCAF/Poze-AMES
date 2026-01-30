import torch 
import logging
import numpy as np 
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data, Batch
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

log_filename = "test.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),  # Log to file (append mode)
        logging.StreamHandler()  # Log to console
    ]
)

# Define the fingerprint generator
class MoleculeProcessor:
    def __init__(self, smiles_list, labels=None, fingerprint_generator=None):
        self.smiles_list = smiles_list
        self.labels = labels
        self.molecules = self._generate_molecules()

    def _generate_molecules(self):
        """
        Generates RDKit Mol objects from SMILES strings and stores them.
        Invalid SMILES are skipped.
        """
        molecules = []
        valid_smiles = []
        
        for smiles in self.smiles_list:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is not None:
                molecules.append(mol)
                valid_smiles.append(smiles)
            else:
                logging.warning("Skipping invalid SMILES: %s", smiles)  

        # Log total number of valid SMILES outside the loop
        logging.debug("Total valid SMILES found: %d", len(valid_smiles))

        self.smiles_list = valid_smiles  # Update the SMILES list to only include valid entries
        return molecules

    def calculate_fingerprint(self, radius=None, fpsize=None):
        """
        Calculates the fingerprint for each molecule.
        """
        fpgen = GetMorganGenerator(radius=radius, fpSize=fpsize, countSimulation=True, includeChirality=True)
        fingerprints = []
        for mol in self.molecules:
            fp = fpgen.GetFingerprint(mol)
            fingerprints.append(np.array(fp.ToList()))
        return np.array(fingerprints)

    def node_edge_feature_extraction(self):
        """
        Extracts node and edge features from the stored molecules for GNN input.
        """
        data_list = []
        for mol, label in zip(self.molecules, self.labels):
            atom_features = []
            edges = []

            # Extract atom features
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetFormalCharge(),
                    atom.GetTotalNumHs(),
                    atom.GetDegree(),
                    int(atom.GetHybridization()),
                    atom.GetIsAromatic(),
                    atom.GetTotalValence(),
                    atom.IsInRing(),
                    atom.GetIsotope()
                ]
                atom_features.append(features)

            node_features = torch.tensor(atom_features, dtype=torch.float32)

            # Extract edge features
            for bond in mol.GetBonds():
                start_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edges.append((start_idx, end_idx))
            edge_index = torch.tensor(edges, dtype=torch.long).t()

            # Create Data object
            data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float32))
            data_list.append(data)

        # Combine into a single batch
        return Batch.from_data_list(data_list)

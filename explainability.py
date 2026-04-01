"""
Explainability module for Poze-AMES.
Adds GNNExplainer, Integrated Gradients, structural alert analysis,
and quantitative comparison tools to complement the existing GAT attention approach.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from types import SimpleNamespace
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import spearmanr
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
import io


# =============================================================================
# 1. GNNExplainer Wrapper
# =============================================================================

class GNNExplainerWrapper(nn.Module):
    """Wraps CombinedModelGATMLP for torch_geometric Explainer API compatibility.

    The Explainer expects forward(x, edge_index, **kwargs) but CombinedModelGATMLP
    expects forward(data, fingerprints) where data has .x, .edge_index, .batch attributes.
    """
    def __init__(self, combined_model):
        super().__init__()
        self.combined_model = combined_model

    def forward(self, x, edge_index, batch=None, fingerprints=None):
        data = SimpleNamespace(x=x, edge_index=edge_index, batch=batch)
        return self.combined_model(data, fingerprints)


def create_gnnexplainer(model, epochs=200, lr=0.01):
    """Create a configured GNNExplainer instance for the wrapped model."""
    wrapper = GNNExplainerWrapper(model)
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type='model',
        model_config=ModelConfig(
            mode='binary_classification',
            task_level='graph',
            return_type='raw',
        ),
        node_mask_type='attributes',
        edge_mask_type='object',
    )
    return explainer, wrapper


def run_gnnexplainer(explainer, data, fingerprints, device='cpu'):
    """Run GNNExplainer on a single molecule and return atom/edge importance.

    Args:
        explainer: Configured Explainer instance
        data: PyG Data object for a single molecule
        fingerprints: Tensor of shape [1, 4096]
        device: Device to run on

    Returns:
        atom_importance: numpy array of shape [num_atoms]
        edge_importance: numpy array of shape [num_edges]
    """
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    fingerprints = fingerprints.to(device)

    explanation = explainer(
        x=x,
        edge_index=edge_index,
        batch=batch,
        fingerprints=fingerprints,
    )

    node_mask = explanation.node_mask.detach().cpu().numpy()
    atom_importance = np.abs(node_mask).sum(axis=1)

    edge_mask = explanation.edge_mask.detach().cpu().numpy() if explanation.edge_mask is not None else None

    return atom_importance, edge_mask


# =============================================================================
# 2. Integrated Gradients
# =============================================================================

class IntegratedGradientsExplainer:
    """Manual Integrated Gradients implementation for atom-level attribution."""

    def __init__(self, model, n_steps=50):
        self.model = model
        self.n_steps = n_steps

    def explain(self, data, fingerprints, device='cpu'):
        """Compute atom-level importance via Integrated Gradients.

        Args:
            data: PyG Data object for a single molecule
            fingerprints: Tensor of shape [1, 4096]
            device: Device to run on

        Returns:
            atom_importance: numpy array of shape [num_atoms]
        """
        self.model.eval()

        x_input = data.x.clone().to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x_input.size(0), dtype=torch.long, device=device)
        fingerprints = fingerprints.to(device)

        baseline_x = torch.zeros_like(x_input)

        total_gradients = torch.zeros_like(x_input)

        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            interpolated_x = baseline_x + alpha * (x_input - baseline_x)
            interpolated_x = interpolated_x.clone().detach().requires_grad_(True)

            data_interp = SimpleNamespace(
                x=interpolated_x,
                edge_index=edge_index,
                batch=batch,
            )

            output = self.model(data_interp, fingerprints)
            output = output.sum()
            output.backward()

            total_gradients += interpolated_x.grad.detach()

        avg_gradients = total_gradients / (self.n_steps + 1)
        attributions = (x_input - baseline_x) * avg_gradients

        atom_importance = attributions.sum(dim=1).abs().detach().cpu().numpy()
        return atom_importance


# =============================================================================
# 3. GAT Attention Extraction (refactored from inference.ipynb)
# =============================================================================

def extract_gat_attention(model, data, fingerprints, device='cpu'):
    """Extract GAT attention weights for a single molecule.

    Returns:
        atom_importance: numpy array of shape [num_atoms] (attention aggregated to atoms)
        bond_attention: numpy array of shape [num_bonds] (raw edge attention scores)
    """
    model.eval()
    data_batch = SimpleNamespace(
        x=data.x.to(device),
        edge_index=data.edge_index.to(device),
        batch=torch.zeros(data.x.size(0), dtype=torch.long, device=device),
    )
    fingerprints = fingerprints.to(device)

    with torch.no_grad():
        _ = model(data_batch, fingerprints)

    edge_index_att, attention_weights = model.get_attention_weights()
    attention_weights = attention_weights.cpu().mean(dim=1).numpy()
    edge_index_att = edge_index_att.cpu().numpy()

    num_atoms = data.x.size(0)
    num_edges = data.edge_index.size(1)

    bond_attention = attention_weights[:num_edges]

    mol_edge_index = data.edge_index.numpy()
    atom_scores = np.zeros(num_atoms)
    atom_counts = np.zeros(num_atoms)
    for i in range(len(bond_attention)):
        if i < mol_edge_index.shape[1]:
            src, dst = mol_edge_index[0, i], mol_edge_index[1, i]
            atom_scores[src] += bond_attention[i]
            atom_counts[src] += 1
            atom_scores[dst] += bond_attention[i]
            atom_counts[dst] += 1

    atom_counts[atom_counts == 0] = 1
    atom_importance = atom_scores / atom_counts

    return atom_importance, bond_attention


# =============================================================================
# 4. Structural Alert Analyzer
# =============================================================================

# Well-established Ames mutagenicity structural alerts
STRUCTURAL_ALERTS = {
    'Nitro group': '[N+](=O)[O-]',
    'Aromatic amine': '[c][NH2]',
    'Azo group': '[#7]=[#7]',
    'Nitroso': '[NX2]=O',
    'Alkyl halide': '[CX4][Cl,Br,I]',
    'Epoxide': 'C1OC1',
    'Aromatic hydroxylamine': '[OH]Nc',
    'PAH core (naphthalene)': 'c1ccc2ccccc2c1',
    'Aromatic chloride': 'c[Cl]',
    'Sulfonate on aromatic': 'cS(=O)(=O)[OH]',
}


class StructuralAlertAnalyzer:
    """Identifies known mutagenic structural alerts in molecules."""

    def __init__(self, alerts=None):
        self.alerts = alerts or STRUCTURAL_ALERTS
        self.patterns = {}
        for name, smarts in self.alerts.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is not None:
                self.patterns[name] = pat

    def find_alerts(self, smiles):
        """Find structural alerts in a molecule.

        Returns:
            alert_atoms: set of atom indices involved in any alert
            alert_details: dict mapping alert name -> list of atom index tuples
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return set(), {}

        alert_atoms = set()
        alert_details = {}

        for name, pattern in self.patterns.items():
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                alert_details[name] = matches
                for match in matches:
                    alert_atoms.update(match)

        return alert_atoms, alert_details

    def get_alert_vector(self, smiles, num_atoms):
        """Get binary vector indicating which atoms are part of known alerts."""
        alert_atoms, _ = self.find_alerts(smiles)
        vector = np.zeros(num_atoms)
        for idx in alert_atoms:
            if idx < num_atoms:
                vector[idx] = 1.0
        return vector


# =============================================================================
# 5. Explanation Comparator
# =============================================================================

class ExplanationComparator:
    """Quantitative comparison of different explanation methods."""

    @staticmethod
    def spearman_correlation(scores_a, scores_b):
        """Spearman rank correlation between two atom importance vectors."""
        if len(scores_a) < 3:
            return float('nan')
        corr, pval = spearmanr(scores_a, scores_b)
        return corr

    @staticmethod
    def topk_jaccard(scores_a, scores_b, k_frac=0.25):
        """Jaccard similarity between top-k most important atoms."""
        k = max(1, int(len(scores_a) * k_frac))
        top_a = set(np.argsort(scores_a)[-k:])
        top_b = set(np.argsort(scores_b)[-k:])
        if len(top_a | top_b) == 0:
            return 0.0
        return len(top_a & top_b) / len(top_a | top_b)

    @staticmethod
    def expert_overlap(scores, alert_vector, k_frac=0.25):
        """Overlap between top-k important atoms and structural alert atoms."""
        alert_atoms = set(np.where(alert_vector > 0)[0])
        if len(alert_atoms) == 0:
            return float('nan')
        k = max(1, int(len(scores) * k_frac))
        top_atoms = set(np.argsort(scores)[-k:])
        if len(top_atoms | alert_atoms) == 0:
            return 0.0
        return len(top_atoms & alert_atoms) / len(top_atoms | alert_atoms)

    @staticmethod
    def fidelity_plus(model, data, fingerprints, atom_importance, k_frac=0.25, device='cpu'):
        """Fidelity+: GNN-branch output change when masking top-k important atoms.

        Measures change in the GNN branch output specifically (before fusion),
        since the fingerprint MLP branch dominates the combined model output
        and is unaffected by node feature masking.
        """
        model.eval()
        k = max(1, int(len(atom_importance) * k_frac))
        top_k = set(np.argsort(atom_importance)[-k:])

        x_orig = data.x.clone().to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x_orig.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            gnn_orig = model.gnn(x_orig, edge_index, batch)

        x_masked = x_orig.clone()
        for idx in top_k:
            x_masked[idx] = 0.0

        with torch.no_grad():
            gnn_masked = model.gnn(x_masked, edge_index, batch)

        # Relative change in GNN output norm
        orig_norm = gnn_orig.norm().item()
        if orig_norm < 1e-10:
            return 0.0
        return (gnn_orig - gnn_masked).norm().item() / orig_norm

    def compare_all(self, methods_scores, alert_vector=None):
        """Compare all pairs of explanation methods.

        Args:
            methods_scores: dict of {method_name: atom_importance_array}
            alert_vector: optional binary vector for structural alerts

        Returns:
            results: dict with comparison metrics
        """
        results = {}
        method_names = list(methods_scores.keys())

        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                name_a, name_b = method_names[i], method_names[j]
                scores_a = methods_scores[name_a]
                scores_b = methods_scores[name_b]

                pair_key = f"{name_a} vs {name_b}"
                results[pair_key] = {
                    'spearman': self.spearman_correlation(scores_a, scores_b),
                    'topk_jaccard': self.topk_jaccard(scores_a, scores_b),
                }

        if alert_vector is not None:
            for name, scores in methods_scores.items():
                results[f"{name}_expert_overlap"] = self.expert_overlap(scores, alert_vector)

        return results


# =============================================================================
# 6. Visualization Helpers
# =============================================================================

def _create_colormap(predicted_label, strength=1.0):
    """Create a custom colormap based on prediction label."""
    light_red = np.array([1.0, 0.6, 0.6])
    light_green = np.array([0.6, 1.0, 0.6])
    red = np.array([1.0, 0.0, 0.0])
    green = np.array([0.0, 1.0, 0.0])

    def blend(c1, c2, t):
        return c1 * (1 - t) + c2 * t

    if predicted_label == 0:
        base_low = blend(light_red, red, strength)
        base_high = blend(light_green, green, 1.0)
    else:
        base_low = blend(light_green, green, strength)
        base_high = blend(light_red, red, 1.0)

    return mcolors.LinearSegmentedColormap.from_list('custom', [base_low, base_high], N=256)


def visualize_atom_importance(smiles, atom_scores, predicted_label, actual_label,
                              method_name="", bond_scores=None, ax=None):
    """Visualize atom importance on a molecule structure.

    Args:
        smiles: SMILES string
        atom_scores: array of per-atom importance scores
        predicted_label: 0 or 1
        actual_label: 0 or 1
        method_name: name of the explanation method
        bond_scores: optional array of per-bond importance scores
        ax: matplotlib axes (if None, creates new figure)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    num_atoms = len(mol.GetAtoms())
    num_bonds = len(mol.GetBonds())

    atom_scores = np.array(atom_scores[:num_atoms], dtype=float)
    atom_scores = np.power(atom_scores, 2) if atom_scores.max() > 0 else atom_scores

    cmap = _create_colormap(predicted_label)

    atom_max = atom_scores.max() if atom_scores.max() > 0 else 1.0
    atom_norm = plt.Normalize(vmin=0, vmax=atom_max)

    atom_color_dict = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx < len(atom_scores):
            atom_color_dict[idx] = cmap(atom_norm(atom_scores[idx]))

    bond_color_dict = {}
    if bond_scores is not None and len(bond_scores) > 0:
        bond_scores = np.array(bond_scores[:num_bonds], dtype=float)
        bond_max = bond_scores.max() if bond_scores.max() > 0 else 1.0
        bond_min = bond_scores.min()
        if bond_min == bond_max:
            bond_max = bond_min + 1e-6
        bond_norm = plt.Normalize(vmin=bond_min, vmax=bond_max)
        for bond in mol.GetBonds():
            idx = bond.GetIdx()
            if idx < len(bond_scores):
                bond_color_dict[idx] = cmap(bond_norm(bond_scores[idx]))

    drawer = Draw.MolDraw2DCairo(400, 350)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_color_dict.keys()),
        highlightAtomColors=atom_color_dict,
        highlightBonds=list(bond_color_dict.keys()) if bond_color_dict else [],
        highlightBondColors=bond_color_dict if bond_color_dict else {},
    )
    drawer.FinishDrawing()
    img = drawer.GetDrawingText()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5))

    img_array = plt.imread(io.BytesIO(img))
    ax.imshow(img_array)
    ax.axis('off')

    label_map = {0: "Non-Mut", 1: "Mutagenic"}
    correct = "Correct" if predicted_label == actual_label else "Wrong"
    ax.set_title(f'{method_name}\nPred: {label_map[predicted_label]} | Act: {label_map[actual_label]} ({correct})',
                 fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=atom_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, fraction=0.03,
                 label='Atom Importance')

    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def visualize_comparison(smiles, methods_dict, predicted_label, actual_label,
                         alert_atoms=None, save_path=None):
    """Side-by-side comparison of multiple explanation methods.

    Args:
        smiles: SMILES string
        methods_dict: dict of {method_name: {'atom_scores': array, 'bond_scores': array or None}}
        predicted_label: 0 or 1
        actual_label: 0 or 1
        alert_atoms: optional set of atom indices for structural alerts
        save_path: optional path to save the figure
    """
    n_methods = len(methods_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5.5))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, scores) in zip(axes, methods_dict.items()):
        visualize_atom_importance(
            smiles,
            scores.get('atom_scores', np.array([])),
            predicted_label,
            actual_label,
            method_name=name,
            bond_scores=scores.get('bond_scores'),
            ax=ax,
        )

    fig.suptitle(f'Explainability Comparison: {smiles[:60]}...', fontsize=10, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
import io


from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
import io


def visualize_alert_overlay(smiles, alert_atoms, method_name="", save_path=None):
    """
    Visualize molecule with ONLY structural alert atoms highlighted in blue.
    All other atoms remain uncolored.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES")
        return

    # Prepare highlight dictionaries
    atom_color_dict = {}
    atom_radii = {}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in alert_atoms:
            # Blue highlight for alert atoms
            atom_color_dict[idx] = (0.3, 0.5, 1.0, 0.8)
            atom_radii[idx] = 0.5

    # Draw molecule
    drawer = Draw.MolDraw2DCairo(450, 400)
    opts = drawer.drawOptions()
    opts.highlightRadius = 0.4

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_color_dict.keys()),
        highlightAtomColors=atom_color_dict,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()

    # Convert image
    img = drawer.GetDrawingText()
    img_array = plt.imread(io.BytesIO(img))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img_array)
    ax.axis('off')
    ax.set_title(f'{method_name} (blue = structural alert)', fontsize=10)

    plt.tight_layout()

    # Save if needed
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
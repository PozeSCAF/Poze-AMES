# Poze-AMES 🧪🧠

**Explaining Ames Mutagenicity: A Hybrid GNN–DNN Approach to Prediction and Feature Interpretation**

Poze-AMES is a deep learning framework for **Ames mutagenicity prediction** that combines **Graph Neural Networks (GNNs)** with **Deep Neural Networks (DNNs)**.  
The model leverages molecular graph representations to achieve strong predictive performance while providing **interpretable feature-level explanations**, which are crucial for cheminformatics and drug discovery workflows.

---

## ✨ Highlights

- Hybrid **GNN + DNN** architecture  
- Molecular graph–based learning  
- Feature and substructure-level explainability  
- Notebook-based **training and inference**  
- Fully reproducible **Conda environment**

---

## ⚙️ Installation

### 1️⃣ Create and Activate the Conda Environment

```bash
conda env create -f env.yaml -n Poze-AMES
conda activate Poze-AMES
```

Verify:
```bash
python --version
```

---

### 2️⃣ Add the Environment as a Jupyter Kernel

This step is required to run the notebooks correctly.

```bash
conda activate Poze-AMES
conda install -y ipykernel

python -m ipykernel install \
  --user \
  --name Poze-AMES \
  --display-name "Python (Poze-AMES)"
```

Check that the kernel is registered:
```bash
jupyter kernelspec list
```

---

## ▶️ Running the Notebooks

1. Start Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```

2. Open a notebook from the `notebooks/` directory  
3. Select the kernel:
```
Kernel → Change Kernel → Python (Poze-AMES)
```

---

## 🏋️ Training

Model training is performed using the provided notebook:

- **`training.ipynb`**
  - Data loading and preprocessing  
  - Model architecture definition  
  - Training and validation  

All hyperparameters and configurations are documented inside the notebook.


---
Please download the model from [link](https://drive.google.com/file/d/1tMb4Bo6vJOixvh5azaiPYv235aOJmyN7/view?usp=drive_link) and use it in the inference code.
---

## 🔍 Inference

Prediction and evaluation are handled via:

- **`inference.ipynb`**
  - Load trained models  
  - Run inference on new molecular inputs  
  - Output mutagenicity predictions  

---

## 🧠 Explainability & Analysis

- Feature importance analysis  
- Interpretation of molecular substructures  
- Visualization of GNN-driven explanations  

Dedicated notebooks are provided for detailed model interpretation.

---

## 📦 Dependencies

All dependencies are specified in:
- `env.yaml` (recommended)

---

## 📝 Notes

- Always ensure the **Python (Poze-AMES)** kernel is selected (not `base`)
- GPU acceleration is used automatically if available
- Results may vary depending on dataset split and random seed


## 📚 Citation / Contact

If you use Poze-AMES in your research, please cite the project appropriately.  
For issues or questions, feel free to open a GitHub issue.

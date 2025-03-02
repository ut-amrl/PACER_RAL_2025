# PACER_RAL_2025
PACER Code Release

**Paper:**  
[Preference Conditioned All-Terrain Costmap Generation](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://arxiv.org/abs/2410.23488&ved=2ahUKEwjrl4vRueyLAxVkjYkEHU0EIL8QFnoECBcQAQ&usg=AOvVaw26ng9Pf509-6-JtVmI50ro)

## Setup Instructions

### 1. Create a New Conda Environment and Install Required Packages

To set up the environment and install dependencies, run the following commands:

```bash
# Create the conda environment
conda create -n pacer-env python=3.8

# Activate the conda environment
conda activate pacer-env

# Install the required packages from requirements.txt
pip install -r requirements.txt
```
## Dataset

### Download the Dataset

Download the dataset from Hugging Face:

[Download PACER Dataset](https://huggingface.co/datasets/ut-amrl/pacer)

Set the `'dataset_root_dir'` in the config/context_data.yaml and config/pref_config.yaml  files to the directory where the dataset is located.

## Context Dataset

The **Context Dataset** contains the context data. To visualize examples from this dataset, run:

```bash
python scripts/context_dataset.py --paths_file <paths.yaml>
```
#### Arguments:
- `--paths_file` (type: `str`, default: `paths.yaml`): Path to the paths config.

## PACER Dataset

The **PACER Dataset** contains a ContextDataset, BEV images, target costmaps, and optionally synthetic terrains for augmentation. Run python pacer_dataset.py to visualize examples from this dataset.

```bash
python scripts/pacer_dataset.py --paths_file <path_to_paths.yaml>
```
#### Arguments:
- `--paths_file` (type: `str`, default: `paths.yaml`): Path to the paths config.

## Training

Set the `'model_save_dir'` in `scripts/paths.yaml` to the path where model checkpoints will be saved.

Performance may be improved if the context encoder model is first trained as a VAE for reconstruction, then having the encoder weights loaded into the PACER model.

```bash
    python scripts/train.py  --paths_file <path_to_paths.yaml> --stage <training_stage_number>
```
#### Arguments:
- `--paths_file` (type: `str`, default: `paths.yaml`): Path to the paths config.
- `--stage` (type: int, default: 1): Training stage (1, 2, or 3).
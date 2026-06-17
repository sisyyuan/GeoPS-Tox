# TOXMuSG

TOXMuSG is a substructure-aware multi-task learning framework for molecular toxicity prediction. It integrates whole-molecule geometric representation learning, BRICS-based substructure decomposition, GIN-based substructure encoding, and attention-based substructure weighting for interpretable multi-organ toxicity prediction.

## Overview

The model follows the pipeline below:

```text
SMILES / molecule
  -> RDKit 3D conformer generation
  -> whole-molecule geometry encoder, LEFTNet / E(3)-GNN
  -> BRICS substructure decomposition
  -> GIN substructure encoder
  -> global-to-substructure attention
  -> weighted substructure representation
  -> endpoint-specific prediction heads
```

The default multi-task endpoints are:

- Cardiotoxicity
- Hepatotoxicity
- Eye irritation
- Respiratory toxicity

Each molecule is assigned an endpoint tag, and the shared molecular representation is routed to the corresponding task-specific prediction head.

## Repository Structure

```text
TOXMuSG/
  configs/
    molnet.yml
    config/
      encoder_bbbp.json
      subencoder_bbbp.json
  dataset/
    master.csv
    toxic_all/
    sub_preprocess/toxic_all/
  modules/
    Datasets.py
    DataLoad.py
    GNNConv.py
    GNNs.py
    chemistryProcess.py
    getSubstructure.py
    model.py
    utils.py
  preprocess/
    preprocess.py
  graph_level_pretrain_all.py
```

## Installation

Python 3.9 is recommended. PyTorch, PyTorch Geometric, and torch-scatter should be installed with mutually compatible CUDA or CPU builds.

```bash
conda create -n toxmusg python=3.9 -y
conda activate toxmusg
pip install -r requirements.txt
```

If `torch-geometric` or `torch-scatter` cannot be installed directly from `requirements.txt`, install the versions matching your PyTorch and CUDA environment from the official PyTorch Geometric wheel index.

## Data Preparation

The default dataset is `toxic_all`. The raw toxicity files, molecular mapping file, and precomputed BRICS substructures are organized under:

```text
dataset/toxic_all/
dataset/sub_preprocess/toxic_all/
```

To regenerate BRICS substructures:

```bash
python preprocess/preprocess.py --dataset toxic_all --method brics --num_workers 4
```

## Training

Run training with the default configuration:

```bash
python graph_level_pretrain_all.py --config configs/molnet.yml
```

The main configuration options are in `configs/molnet.yml`, including dataset name, encoder configuration, subencoder configuration, batch size, learning rate, epoch number, decomposition method, and device.

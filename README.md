[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs44286--026--00361--8-blue)](https://doi.org/10.1038/s44286-026-00361-8)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python package](https://github.com/LopezGroup-ICIQ/gamenet_uq/actions/workflows/tests.yml/badge.svg)](https://github.com/LopezGroup-ICIQ/gamenet_uq/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/gamenet-uq.svg)](https://pypi.org/project/gamenet-uq/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/gamenet-uq?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/gamenet-uq)
[![codecov](https://codecov.io/gh/LopezGroup-ICIQ/gamenet_uq/graph/badge.svg?token=W1GBOYU6Q0)](https://codecov.io/gh/LopezGroup-ICIQ/gamenet_uq)

# GAME-Net-UQ

<div style="display: flex; justify-content: center; align-items: center;">
    <p align="center">
     <img src="https://raw.githubusercontent.com/LopezGroup-ICIQ/gamenet_uq/main/GNN_github.png" width="90%" height="90%" />
    </p>
</div>

This repository contains the Python code used to train and evaluate GAME-Net-UQ, a graph neural network with uncertainty quantification (UQ) for predicting the DFT energy of relaxed species and transition states adsorbed on metal surfaces. 

## Install

```bash
pip install gamenet-uq
```

The main dependencies of the repo can be found in [pyproject.toml](./pyproject.toml)

## Dataset

The datasets used to develop GAME-Net-UQ can be found in the Zenodo repository [https://doi.org/10.5281/zenodo.17977395](https://doi.org/10.5281/zenodo.17977395).

1. The ASE database (212 MB) containing all relaxed DFT structures and transition states is located at `gamenetuq_results/ASE_database/all.db`. 
2. The final, clean graph dataset (102 MB) in PyG format is located at `gamenetuq_results/PyG_graph_database/all_scaled_energy_025_125_2_False_False_False_True_False`. The graph dataset can be recreated from the ASE database with the [gen_dataset.py](./scripts/gen_dataset.py) script.  

## Model training and finetuning

To train the model, run the script [train_mve.py](./scripts/train_mve.py). The [input template ](./scripts/input.toml) file provides an explanation for each entry required in the training configuration file.

```bash
python train_mve.py -i input.toml -o output_dirname
```

## Pretrained model

The final pretrained model can be employed with CARE ([link](https://github.com/LopezGroup-ICIQ/care)). 

## License

The code is released under the [MIT](./LICENSE) license.

## Reference

Morandi, S., Loveday, O., Renningholtz, T. *et al.* An end-to-end framework for reactivity in heterogeneous catalysis. *Nat. Chem. Eng.* (2026). [https://doi.org/10.1038/s44286-026-00361-8](https://doi.org/10.1038/s44286-026-00361-8)

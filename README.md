# Step Forward Cross Validation for Bioactivity Prediction

This repo contains code to reproduce the results of our [SFCV Paper]().
These results include model predictions, tables, and images.
Efforts are made to ensure reproducibility of this project.
In case of undefined behaviour or errors in installing or benchmarking, please open an issue.

## Environment Setup

This project uses [pyvenv](https://docs.python.org/3/library/venv.html) to manage python
environment with `Python 3.11`. The following command will create virtual env in `.venv` directory.

### Create Venv

```shell
python3.11 -m venv .venv
```

### Install Requirements

```shell
pip install -r requirements.txt
```

## Dataset

Landrum &
Riniker [[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049) | [Data](https://github.com/rinikerlab/overlapping_assays/tree/main/datasets/source_data)]

### 1. Download Datasets and Standardize SMILES

- Please open and run [**00_Data_source_and_standardize.ipynb
  **](https://github.com/Manas02/sfcv/blob/main/notebook/00_Data_source_and_standardize.ipynb) to download the
above-mentioned dataset and to standardize the SMILES in those files.

### 2. Predicting LogP, LogD and Computing MCE-18

- Follow that with running [**01_Data_add_LogP_LogD_MCE18.ipynb
  **](https://github.com/Manas02/sfcv/blob/main/notebook/01_Data_add_LogP_LogD_MCE18.ipynb)
to predict and add data for
CrippenLogP ([rdkit](https://www.rdkit.org/docs/GettingStartedInPython.html#descriptor-calculation)),
LogD ([Code](https://gist.github.com/PatWalters/7aebcd5b87ceb466db91b11e07ce3d21)) and
compute [MCE-18](https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.9b00004).

### 3. Comparing the changes in number of compounds after standardization and deduplication

- Follow this with running [**02_Table_mol_per_target_before_after_standardization.ipynb
  **](https://github.com/Manas02/sfcv/blob/main/notebook/02_Table_mol_per_target_before_after_standardization.ipynb)
  to generate the table and parity plot. The results are saved in `benchmark/results/tables` and
  `benchmark/results/figures` directories. 

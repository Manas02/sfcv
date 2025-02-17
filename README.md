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

Please open and
  run [00_Data_source_and_standardize.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/00_Data_source_and_standardize.ipynb)
  to download the
above-mentioned dataset and to standardize the SMILES in those files.

### 2. Predicting LogP, LogD and Computing MCE-18

Follow that by
  running [01_Data_add_LogP_LogD_MCE18.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/01_Data_add_LogP_LogD_MCE18.ipynb)
to predict and add data for
CrippenLogP ([rdkit](https://www.rdkit.org/docs/GettingStartedInPython.html#descriptor-calculation)),
LogD ([Code](https://gist.github.com/PatWalters/7aebcd5b87ceb466db91b11e07ce3d21)) and
compute [MCE-18](https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.9b00004).

### 3. Comparing the changes in number of compounds after standardization and deduplication

Follow this with
  running [02_Table_mol_per_target_before_after_standardization.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/02_Table_mol_per_target_before_after_standardization.ipynb)
  to generate the table and parity plot. The results are saved in `benchmark/results/tables` and
  `benchmark/results/figures` directories. 

### 4. Comparing and Plotting the Distributions of Properties in Dataset

Run [03_Plots_Table_target_properties.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/03_Plots_Table_target_properties.ipynb)
to get the summary of properties as a table and to plot the distributions.

## Method: Data Splitting

### 1. Implementing `SortedStepForwardCV` and `UnsortedStepForwardCV`

Run [04_Implementation_SFCV.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/04_Implementation_SFCV.ipynb)
  to visualise how SortedStepForwardCV and UnsortedStepForwardCV work.

### 2. Implementing `ScaffoldSplitCV`

Run [05_Implementation_ScaffoldSplitCV.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/05_Implementation_ScaffoldSplitCV.ipynb)
to check how ScaffoldSplitCV works. The algorithm groups molecules by their chemical scaffolds, shuffles these groups,
and sequentially assigns entire scaffold groups to the training set until a target fraction is reached, with the
remaining groups forming the test set.

### 3. Implementing `RandomSplitCV`

Run [06_Implementation_RandomSplitCV.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/06_Implementation_RandomSplitCV.ipynb)
to check how RandomSplitCV works.

### 4. Plotting Chemical Space wrt Split Type

Run [08_Plots_chemical_space_across_split.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/08_Plots_chemical_space_across_split.ipynb)
to visualise chemical space wrt Split types.
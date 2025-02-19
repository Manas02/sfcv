# Step Forward Cross Validation for Bioactivity Prediction

This repo contains code to reproduce the results of [SFCV Paper]().
These results include model predictions, tables, and images.
Efforts are made to ensure reproducibility of this project.
In case of undefined behaviour or errors in installing or benchmarking, please open an issue.
---
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

---
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

---
# Method

## 1. Data Splitting

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

### 4. Validating the Splits produce (almost) equal number of test compounds per fold

Run [07_Validate_train_test_split.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/07_Validate_train_test_split.ipynb)
to visualise number of molecules in test set across folds across targets.

### 5. Plotting Chemical Space wrt Split Type

Run [08_Plots_chemical_space_across_split.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/08_Plots_chemical_space_across_split.ipynb)
to visualise chemical space wrt Split types.

### 6. Plotting & Comparing Distribution of Sorting properties per Split type per Fold across Targets

Run [09_Plots_Table_split_properties.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/09_Plots_Table_split_properties.ipynb)
to visualise distributions of sorting properties wrt Split types per fold averaged over all targets.

---
## 2. Metrics

### 1. Implementing Discovery Yield

Run [10_Implimentation_Discovery_Yield.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/10_Implimentation_Discovery_Yield.ipynb)
to understand and visualise the illustrative example of discovery yield.

### 2. Implementing Novelty Error

Run [11_Implimentation_Novelty_Error.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/11_Implimentation_Novelty_Error.ipynb)
to understand and visualise the illustrative example of novelty error.

### 3. Implementing Benchmark

Run [12_Implementation_Benchmark.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/12_Implementation_Benchmark.ipynb)
to see how benchmarking was performed.

---

## 3. Results

### 1. Extract Results

Run [13_Table_extract_results.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/13_Table_extract_results.ipynb)
to extract results into digestable format.

### 2. Plot Results

Run [14_Plots_results.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/14_Plots_results.ipynb),
[15_Plots_Result_hERG.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/15_Plots_Result_hERG.ipynb) and
[16_Plots_Result_MAPK.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/16_Plots_Result_MAPK.ipynb),
[17_Plots_Result_VEGFR.ipynb](https://github.com/Manas02/sfcv/blob/main/notebook/17_Plots_Result_VEGFR.ipynb) to
visualise the results.
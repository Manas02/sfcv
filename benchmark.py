# Step Forward Cross Validation for Bioactivity Prediction
## Benchmark for hERG, MAP14K and VEGFR2 for 3 fingerprints (ECFP4, RDKit and AtomPair)

import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from sfcv.datasplit import (
    SortedStepForwardCV,
    UnsortedStepForwardCV,
    ScaffoldSplitCV,
    RandomSplitCV,
)

### Fingerprint Calculation

ecfp4gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)


def compute_ecfp4(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return ecfp4gen.GetFingerprintAsNumPy(mol)


def compute_rdkit_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return rdkgen.GetFingerprintAsNumPy(mol)


def compute_atompair_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return apgen.GetFingerprintAsNumPy(mol)


#### Since, we'll be training on these fingerprints, precomputing these fingerprints and saving them will save some time.

molecule_set = set()

for fname in os.listdir("./benchmark/data/processed"):
    if fname.endswith(".csv"):
        df = pd.read_csv(f"./benchmark/data/processed/{fname}")
        molecule_set |= set(df["standardized_smiles"].unique())

len(molecule_set)

smi2ecfp4 = {}
smi2atompair = {}
smi2rdkit = {}

for smi in tqdm(molecule_set, desc="Computing Fingerprints"):
    smi2ecfp4[smi] = compute_ecfp4(smi)
    smi2atompair[smi] = compute_atompair_fp(smi)
    smi2rdkit[smi] = compute_rdkit_fp(smi)

## Saving the split columns

os.makedirs("./benchmark/data/final/", exist_ok=True)

cv_splitters = {
    "RandomSplit": RandomSplitCV(frac_train=0.9, n_folds=10, seed=69420),
    "ScaffoldSplit": ScaffoldSplitCV(
        smiles_col="standardized_smiles",
        n_folds=10,
        frac_train=0.9,
        seed=69420,
        include_chirality=False,
    ),
    "SortedStepForward_LogD": SortedStepForwardCV(
        sorting_col="LogD", ideal=2, n_bins=10, ascending=False
    ),
    "SortedStepForward_LogP": SortedStepForwardCV(
        sorting_col="LogP", ideal=2, n_bins=10, ascending=False
    ),
    "SortedStepForward_MCE18": SortedStepForwardCV(
        sorting_col="MCE18", n_bins=10, ascending=True
    ),
    "UnsortedStepForward": UnsortedStepForwardCV(n_bins=10, random_state=69420),
}


def add_cv_split_columns(df, cv_splitters):
    df = df.copy()
    for split_name, cv_splitter in cv_splitters.items():
        for fold_idx, (train_idx, test_idx) in enumerate(
            cv_splitter.split(df), start=1
        ):
            col_name = f"{split_name}_Fold_{fold_idx}"
            df[col_name] = None
            df.loc[train_idx, col_name] = "Train"
            df.loc[test_idx, col_name] = "Test"
    return df


for fname in tqdm(os.listdir("./benchmark/data/processed/"), desc="Processing Splits"):
    if os.path.exists(f"./benchmark/data/final/{fname}"):
        continue
    if fname.endswith(".csv"):
        df = pd.read_csv(f"./benchmark/data/processed/{fname}")
        df = add_cv_split_columns(df, cv_splitters)
        df.to_csv(f"./benchmark/data/final/{fname}")


## Models


def mlp_regressor_factory(n_train, random_state=42):
    n_hidden = min(25, int(np.sqrt(n_train)))
    return MLPRegressor(
        hidden_layer_sizes=(n_hidden,), random_state=random_state, max_iter=1000
    )


def xgb_regressor_factory(n_train, random_state=42):
    n_estimators = min(25, int(np.sqrt(n_train)))
    return XGBRegressor(n_estimators=n_estimators, random_state=random_state)


def rf_regressor_factory(n_train, random_state=42):
    n_trees = min(25, int(np.sqrt(n_train)))
    return RandomForestRegressor(n_estimators=n_trees, random_state=random_state)


regressor_factories = [
    rf_regressor_factory,
    xgb_regressor_factory,
    mlp_regressor_factory,
]


def process_regressor(regressor_factory, X_train, y_train, fingerprint_vals):
    regressor = regressor_factory(len(X_train))
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(np.vstack(fingerprint_vals))
    identifier = getattr(regressor_factory, "__name__", str(regressor_factory))
    return identifier, y_pred


## Bulk Tanimoto Similarity


def bulk_tanimoto_similarity(mol_fp: np.ndarray, list_of_fps: np.ndarray) -> np.ndarray:
    intersection = np.sum(list_of_fps & mol_fp, axis=1)
    union = np.sum(list_of_fps | mol_fp, axis=1)
    return intersection / union


## Let's Train the models

os.makedirs("./benchmark/data/results/", exist_ok=True)

fp2map = {"ECFP4": smi2ecfp4, "RDKitFP": smi2rdkit, "AtomPairsFP": smi2atompair}

# Initialize a dictionary to hold new columns
new_columns = {}

for fname in tqdm(os.listdir("./benchmark/data/final/"), desc="Training"):
    if fname.endswith(".csv"):
        df = pd.read_csv(f"./benchmark/data/final/{fname}")

        fold_cols = [i for i in df.columns if "_Fold_" in i]
        for fold_col in tqdm(fold_cols, desc="Processing Splits and Folds"):
            test_mask = df[fold_col] == "Test"
            train_mask = df[fold_col] == "Train"

            for fp, mapping in fp2map.items():
                df[fp] = df["standardized_smiles"].map(mapping)
                test_fps = np.vstack(df.loc[test_mask, fp].values)
                train_fps = np.vstack(df.loc[train_mask, fp].values)
                max_tcs = [
                    max(bulk_tanimoto_similarity(test_fp, train_fps))
                    for test_fp in test_fps
                ]
                tc_series = pd.Series(data=max_tcs, index=df.loc[test_mask].index)
                new_columns[f"{fold_col}_{fp}_Tc"] = tc_series
                X_train = np.vstack(df.loc[train_mask, fp].values)
                y_train = df.loc[train_mask, "pchembl_value"].values
                X = np.vstack(df[fp].values)

                for regressor_factory in regressor_factories:
                    model_name, preds = process_regressor(
                        regressor_factory, X_train, y_train, X
                    )
                    new_col_name = f"{fold_col}_{fp}_{model_name}"
                    new_columns[new_col_name] = preds

        if new_columns:
            new_cols_df = pd.DataFrame(new_columns, index=df.index)
            df = pd.concat([df, new_cols_df], axis=1)

        df.to_csv(f"./benchmark/data/results/{fname}")

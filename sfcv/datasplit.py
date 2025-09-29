from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class SortedStepForwardCV:
    def __init__(self, sorting_col, ideal=None, n_bins=10, ascending=True):
        self.sorting_col = sorting_col
        self.ideal = ideal
        self.n_bins = n_bins
        self.ascending = ascending

    def _sort_indices(self, df):
        if self.ideal is not None:
            # Compute the absolute distance from the ideal value.
            distances = (df[self.sorting_col] - self.ideal).abs()
            # Stable sort based on distance
            sorted_idx = np.argsort(distances, kind="stable")
        else:
            # Use the property value directly, taking into account the desired order.
            sorted_idx = np.argsort(df[self.sorting_col].values, kind="stable")
        if not self.ascending:
            sorted_idx = sorted_idx[::-1]
        return sorted_idx

    def _create_bins(self, sorted_idx):
        n = len(sorted_idx)
        # Determine bin sizes: use np.array_split to handle duplicates and non-divisible sizes gracefully.
        bins = np.array_split(sorted_idx, self.n_bins)
        return bins

    def split(self, df):
        sorted_idx = self._sort_indices(df)
        bins = self._create_bins(sorted_idx)
        n_bins = len(bins)

        for i in range(1, n_bins):
            train_idx = np.concatenate(bins[:i])
            test_idx = bins[i]
            yield train_idx, test_idx


class UnsortedStepForwardCV:
    def __init__(self, n_bins=10, random_state=69420):
        self.n_bins = n_bins
        self.random_state = random_state

    def _shuffle_indices(self, df):
        indices = np.arange(len(df))
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(indices)
        return indices

    def _create_bins(self, shuffled_idx):
        bins = np.array_split(shuffled_idx, self.n_bins)
        return bins

    def split(self, df):
        shuffled_idx = self._shuffle_indices(df)
        bins = self._create_bins(shuffled_idx)
        n_bins = len(bins)

        for i in range(1, n_bins):
            train_idx = np.concatenate(bins[:i])
            test_idx = bins[i]
            yield train_idx, test_idx


class ScaffoldSplitCV:
    def __init__(self, smiles_col="standardized_smiles", n_folds=10, frac_train=0.9, seed=69420,
                 include_chirality=False):
        self.smiles_col = smiles_col
        self.frac_train = frac_train
        self.seed = seed
        self.n_folds = n_folds
        self.include_chirality = include_chirality

    def split(self, df):
        smiles_list = df[self.smiles_col].tolist()
        for i in range(1, self.n_folds):
            yield self._scaffold_split(smiles_list, self.frac_train, i * self.seed, self.include_chirality)

    def _scaffold_split(self, smiles_list, frac_train, seed, include_chirality):
        scaffold_to_indices = defaultdict(list)
        for idx, smiles in enumerate(smiles_list):
            scaffold = self._generate_scaffold(smiles, include_chirality)
            scaffold_to_indices[scaffold].append(idx)

        scaffold_groups = list(scaffold_to_indices.values())
        rng = np.random.RandomState(seed)
        rng.shuffle(scaffold_groups)

        n_total = len(smiles_list)
        n_train = int(np.floor(frac_train * n_total))

        train_indices = []
        test_indices = []

        for group in scaffold_groups:
            if len(train_indices) + len(group) <= n_train:
                train_indices.extend(group)
            else:
                test_indices.extend(group)

        return np.array(train_indices), np.array(test_indices)

    @staticmethod
    def _generate_scaffold(smiles, include_chirality=False):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold


class RandomSplitCV:
    def __init__(self, frac_train=0.9, n_folds=10, seed=69420):
        self.frac_train = frac_train
        self.n_folds = n_folds
        self.seed = seed

    def split(self, df):
        n_total = len(df)
        indices = np.arange(n_total)

        for i in range(1, self.n_folds):
            fold_seed = self.seed * i
            rng = np.random.RandomState(fold_seed)
            shuffled_indices = rng.permutation(indices)

            n_train = int(np.floor(self.frac_train * n_total))
            train_indices = shuffled_indices[:n_train]
            test_indices = shuffled_indices[n_train:]

            yield train_indices, test_indices

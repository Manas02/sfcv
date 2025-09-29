import os
import pathlib

import lightgbm as lgbm
import molvs
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import rdchem, Descriptors

RDLogger.DisableLog("rdApp.*")
MODEL_PATH = pathlib.Path(__file__).parent / "logdmodel.txt"

md = molvs.metal.MetalDisconnector()
lfc = molvs.fragment.LargestFragmentChooser()
uc = molvs.charge.Uncharger()


def standardize_smiles(smiles):
    std_smiles = molvs.standardize.standardize_smiles(smiles)
    std_mol = Chem.MolFromSmiles(std_smiles)
    std_mol = md.disconnect(std_mol)
    std_mol = lfc.choose(std_mol)
    std_mol = uc.uncharge(std_mol)
    std_smi = Chem.MolToSmiles(std_mol)
    if not molvs.validate.validate_smiles(std_smi):
        std_smi = molvs.standardize.canonicalize_tautomer_smiles(std_smi)
        return std_smi


class LogDPredictor:
    def __init__(self, model_file_name=MODEL_PATH):
        if not os.path.exists(model_file_name):
            raise FileNotFoundError(f"model file not found in {model_file_name}")
        self.mdl = lgbm.Booster(model_file=str(model_file_name))
        self.descList = self.mdl.feature_name()
        self.fns = [(x, y) for x, y in Descriptors.descList if x in self.descList]

    def predict_smiles(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return self.predict(mol)
        return None

    def predict(self, mol):
        res = []
        for _, y in self.fns:
            res.append(y(mol))
        return self.mdl.predict([res])[0]


def compute_mce18(smiles: str) -> float | None:
    # copypasta from https://github.com/SkylerKramer/MedChem/blob/master/mce18.txt
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ar = rdmd.CalcNumAromaticRings(mol)
    nar = rdmd.CalcNumAliphaticRings(mol)
    spiro = rdmd.CalcNumSpiroAtoms(mol)
    sp3 = rdmd.CalcFractionCSP3(mol)
    chiral = int(bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))

    zagreb = 0
    index = 0
    cyc = 0
    for atom in mol.GetAtoms():
        zagreb = zagreb + rdchem.Atom.GetDegree(atom) ** 2
        if (
                atom.GetAtomicNum() == 6
                and mol.GetAtomWithIdx(index).IsInRing() == True
                and rdchem.Atom.GetHybridization(atom) == 4
        ):
            cyc += 1
        index += 1

    cyc = cyc / mol.GetNumAtoms()
    acyc = sp3 - cyc
    q = 3 - 2 * mol.GetNumAtoms() + zagreb / 2
    return q * (ar + nar + spiro + chiral + (sp3 + cyc - acyc) / (1 + sp3))


pred = LogDPredictor()


def predict_logd(smiles: str) -> float | None:
    return pred.predict_smiles(smiles)


def predict_logp(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Descriptors.MolLogP(mol)

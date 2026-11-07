"""Shared loading, featurisation and scoring for the Ames experiments.

The data is the shared 22,157-structure set with its existing train/validation/
test split. The test part is never loaded here. It was already used once for the
ADMET-AI comparison, so selecting anything on it would be selecting on a set we
have already seen; it is reserved for a single final report.

Two groupings are available and they are not the same thing. `leakage_group` is
the grouping the shared work split on, built from parent structures. Bemis-Murcko
scaffolds are the grouping used for the enzyme models in this repository. Which
one is stricter is measurable rather than obvious, so both are offered and the
experiments report which was used.
"""
import collections
import csv
from pathlib import Path
import warnings

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DATA = PROJECT / 'data' / 'ames' / 'Ames_모델용_전체_22157개.csv'
RESULTS = ROOT.parent / 'results'

FOLDS = 5
REPEATS = 2
SEED = 20261106
MIN_TEST = 50

# Descriptors that are the same ten the enzyme models used, kept so the two
# measurement records can be read side by side.
CORE_DESCRIPTORS = ('MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                    'NumRotatableBonds', 'RingCount', 'FractionCSP3',
                    'NumAromaticRings', 'HeavyAtomCount')


def load(splits=('train', 'validation')):
    """Structures, labels and both groupings for the splits named.

    Passing 'test' is allowed but deliberately inconvenient: no experiment in
    this folder should call for it except the single final evaluation.
    """
    if not DATA.is_file():
        raise SystemExit(f'자료를 찾지 못했습니다: {DATA}')
    wanted = set(splits)
    smiles, labels, leakage, kept_rows = [], [], [], []
    seen = set()
    with DATA.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] not in wanted:
                continue
            item = (row.get('canonical_smiles') or '').strip()
            if not item or item in seen:
                continue
            molecule = Chem.MolFromSmiles(item)
            if molecule is None:
                continue
            seen.add(item)
            smiles.append(item)
            labels.append(int(row['label']))
            leakage.append(row['leakage_group'])
            kept_rows.append(row)
    molecules = [Chem.MolFromSmiles(item) for item in smiles]
    return {
        'smiles': smiles, 'molecules': molecules,
        'labels': np.array(labels),
        'leakage_groups': np.array(leakage),
        'scaffold_groups': np.array([scaffold_of(item) for item in smiles]),
        'rows': kept_rows, 'splits': tuple(splits),
    }


def scaffold_of(smiles):
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles,
                                                       includeChirality=False)
    except Exception:
        scaffold = ''
    # An acyclic structure has no Murcko scaffold. Giving them all one empty
    # group would tie thousands of unrelated structures together, so each keeps
    # its own group instead.
    return scaffold or f'acyclic::{smiles}'


def _generator(kind, size):
    if kind == 'rdkit':
        return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=size)
    radius = 3 if kind == 'morgan_r3' else (1 if kind == 'morgan_r1' else 2)
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)


def _fingerprint(molecules, kind, size):
    generator = _generator(kind, size)
    matrix = np.zeros((len(molecules), size), dtype=np.float32)
    for index, molecule in enumerate(molecules):
        for bit in generator.GetFingerprint(molecule).GetOnBits():
            matrix[index, bit] = 1.0
    return matrix


def _maccs(molecules):
    matrix = np.zeros((len(molecules), 167), dtype=np.float32)
    for index, molecule in enumerate(molecules):
        for bit in MACCSkeys.GenMACCSKeys(molecule).GetOnBits():
            matrix[index, bit] = 1.0
    return matrix


def _descriptors(molecules, names=None):
    described = [CalcMolDescriptors(molecule) for molecule in molecules]
    if names is None:
        # Columns that never vary carry nothing and only slow the fit down.
        keys = sorted(described[0])
        matrix = np.array([[float(row.get(key) or 0.0) for key in keys]
                           for row in described], dtype=np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        keep = matrix.std(axis=0) > 1e-9
        return matrix[:, keep]
    matrix = np.array([[float(row.get(name) or 0.0) for name in names]
                       for row in described], dtype=np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _join(*blocks):
    return np.hstack(blocks).astype(np.float32)


# name -> builder. The shared work's best configuration was an RDKit path
# fingerprint plus the full 2D descriptor block, so that one is included under
# the name it is referred to by in the shared report.
FEATURES = {
    'morgan_r2_2048': lambda m: _fingerprint(m, 'morgan_r2', 2048),
    'morgan_r2_1024': lambda m: _fingerprint(m, 'morgan_r2', 1024),
    'morgan_r3_2048': lambda m: _fingerprint(m, 'morgan_r3', 2048),
    'rdkit_fp_2048': lambda m: _fingerprint(m, 'rdkit', 2048),
    'maccs_167': _maccs,
    'descriptors_all': lambda m: _descriptors(m),
    'descriptors_10': lambda m: _descriptors(m, CORE_DESCRIPTORS),
    'morgan_r2_2048+descriptors_10': lambda m: _join(
        _fingerprint(m, 'morgan_r2', 2048), _descriptors(m, CORE_DESCRIPTORS)),
    'morgan_r2_2048+descriptors_all': lambda m: _join(
        _fingerprint(m, 'morgan_r2', 2048), _descriptors(m)),
    'rdkit_fp_2048+descriptors_all': lambda m: _join(
        _fingerprint(m, 'rdkit', 2048), _descriptors(m)),
}

DEFAULT_FEATURES = 'rdkit_fp_2048+descriptors_all'


def models(seed):
    """Builders rather than instances, so every fold starts from a fresh model."""
    built = {
        'random_forest': lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
            class_weight='balanced', random_state=seed),
        'extra_trees': lambda: ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
            class_weight='balanced', random_state=seed),
        'hist_gradient_boosting': lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=seed),
        'logistic_regression': lambda: Pipeline([
            ('scale', StandardScaler(with_mean=False)),
            ('model', LogisticRegression(max_iter=3000, C=0.1,
                                         class_weight='balanced'))]),
        'bernoulli_nb': lambda: BernoulliNB(),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError:
        pass
    else:
        # The shared work's parameters, so our run and theirs differ in the
        # evaluation method rather than in the model.
        built['xgboost'] = lambda: XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.0677977974602176,
            subsample=0.993506466625609, colsample_bytree=0.6552092851888845,
            reg_lambda=0.1, min_child_weight=1.1821791088876834,
            tree_method='hist', n_jobs=-1, random_state=seed,
            eval_metric='logloss')
        # The same model with the class imbalance carried in the training
        # weights instead of left for a decision line to absorb.
        #
        # The collection is 30.2% positive and the shipped model was fitted
        # with weighting off ("balanced": false in its spec), so its scores
        # sit low and 0.5 lands far from the balance point: it reads
        # sensitivity 0.70 against specificity 0.94 there. Published Ames
        # benchmarks are near 50/50, which is why 0.5 works for them and why
        # moving a decision line is not something their papers have to do.
        #
        # scale_pos_weight is the ratio XGBoost documents for this: negatives
        # over positives, here 0.6977 / 0.3023.
        built['xgboost_balanced'] = lambda: XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.0677977974602176,
            subsample=0.993506466625609, colsample_bytree=0.6552092851888845,
            reg_lambda=0.1, min_child_weight=1.1821791088876834,
            scale_pos_weight=POSITIVE_WEIGHT,
            tree_method='hist', n_jobs=-1, random_state=seed,
            eval_metric='logloss')
    return built


# Negatives over positives across the collection (30.2% positive), which is
# what XGBoost's scale_pos_weight expects. Folds vary by a fraction of a
# percent; one figure for the whole collection keeps every fold comparable.
POSITIVE_WEIGHT = 0.6977 / 0.3023

DEFAULT_MODEL = 'xgboost'
BINARY_ONLY = ('bernoulli_nb',)


def splitter(seed, repeat):
    return StratifiedGroupKFold(n_splits=FOLDS, shuffle=True,
                                random_state=seed + repeat)


def grouped_score(matrix, labels, groups, builder, seed, repeats=REPEATS):
    """Mean AUC over grouped folds, with the spread that makes it readable.

    A single split of a set this size still moves by a few hundredths when the
    split changes, and every comparison in these experiments is smaller than
    that, so nothing here is scored on one split.
    """
    aucs = []
    for repeat in range(repeats):
        for train, test in splitter(seed, repeat).split(matrix, labels, groups):
            if len(set(labels[test])) < 2 or len(test) < MIN_TEST:
                continue
            model = builder()
            model.fit(matrix[train], labels[train])
            aucs.append(roc_auc_score(labels[test],
                                      model.predict_proba(matrix[test])[:, 1]))
    if not aucs:
        return None
    return {'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            'folds_scored': len(aucs)}


def noise_floor(matrix, labels, groups, seed, repeats=REPEATS):
    """What the same folds give for a model that learns nothing.

    Without this number a difference of a few hundredths cannot be read, and
    most of the differences in these experiments are that size.
    """
    return grouped_score(matrix, labels, groups,
                         lambda: DummyClassifier(strategy='stratified',
                                                 random_state=seed),
                         seed, repeats)


def group_summary(groups):
    counts = collections.Counter(groups)
    sizes = np.array(sorted(counts.values(), reverse=True))
    return {'groups': int(len(counts)),
            'largest': int(sizes[0]), 'median_size': float(np.median(sizes)),
            'singletons': int((sizes == 1).sum()),
            'share_in_singletons': float(sizes[sizes == 1].sum() / sizes.sum())}

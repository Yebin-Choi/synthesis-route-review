"""AiZynthFinder adapters for the shared research screening rules.

The retained numeric APIs are uncalibrated search heuristics, not toxicity
probabilities, impurity clearance estimates or evidence of safety. Chemistry
and vetted intake references live only in app/toxicology.py.
"""

import math
from pathlib import Path
import sys

from rdkit import Chem

APP_DIR = str(Path(__file__).resolve().parents[1] / "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from toxicology import assess_material

from aizynthfinder.context.policy.filter_strategies import FilterStrategy
from aizynthfinder.context.scoring import Scorer
from aizynthfinder.utils.exceptions import RejectionException


def _assessment(rd_mol):
    if rd_mol is None:
        return assess_material("")
    try:
        smiles = Chem.MolToSmiles(rd_mol, isomericSmiles=True)
    except (TypeError, ValueError, RuntimeError):
        return assess_material("")
    return assess_material(smiles)


def _heuristic_priority(assessment):
    if assessment["status"] in ("invalid", "known_concern") or assessment["special_concern"]:
        return 1.0
    if assessment["status"] == "structural_alert":
        return 0.5
    return 0.0


def find_ich_m7(rd_mol):
    """Return (listed name, vetted reference or None), not an ICH M7 class."""
    assessment = _assessment(rd_mol)
    if assessment["status"] == "known_concern":
        return assessment["name"], assessment["intake_ug_day"]
    return None


def find_alerts(rd_mol):
    """Return (custom alert, uniform heuristic weight) pairs, not potency."""
    return [(label, 0.5) for label in _assessment(rd_mol)["alerts"]]


def risk_of(rd_mol):
    """Uncalibrated search priority: 0 means only 'no custom alert detected'.

    Invalid input, known entries and detected special concern receive 1;
    other custom alerts receive 0.5. These are not risk probabilities.
    """
    return _heuristic_priority(_assessment(rd_mol))


class GenotoxFilter(FilterStrategy):
    """Heuristically reject flagged reactants; always reject invalid input.

    cutoff=1 rejects known entries and detected special concern. cutoff<=0.5
    also rejects other custom alerts. Acceptance is not a safety determination.
    """

    def __init__(self, key, config, **kwargs):
        super().__init__(key, config, **kwargs)
        self.cutoff = float(kwargs.get("cutoff", 1.0))
        if not math.isfinite(self.cutoff) or not 0 < self.cutoff <= 1:
            raise ValueError("cutoff must be finite and greater than 0, at most 1")
        self.rejected = []

    def _reject(self, smiles, reason):
        self.rejected.append((smiles, reason))
        raise RejectionException(f"Research screening heuristic rejected reactant: {reason}")

    def apply(self, reaction):
        try:
            outcomes = reaction.reactants
            index = reaction.index
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                self._reject("", "invalid reactant outcome index; screening unavailable")
            reactants = outcomes[index]
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            self.rejected.append(("", "invalid or unavailable reactant outcome"))
            raise RejectionException(
                "Research screening unavailable for reactant outcome; rejected fail-closed"
            ) from exc
        if not reactants:
            self._reject("", "empty reactant outcome; screening unavailable")

        for mol in reactants:
            try:
                assessment = _assessment(mol.rd_mol)
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                self.rejected.append(("", "invalid reactant structure; screening unavailable"))
                raise RejectionException(
                    "Research screening unavailable for reactant; rejected fail-closed"
                ) from exc
            if assessment["status"] == "invalid":
                self._reject("", "invalid reactant structure; screening unavailable")
            if _heuristic_priority(assessment) >= self.cutoff:
                reason = assessment["name"] or ", ".join(assessment["alerts"])
                self._reject(assessment["canonical_smiles"], reason)


class GenotoxScorer(Scorer):
    """Uncalibrated search preference over non-target route materials.

    Higher means fewer or less heavily weighted screening flags, not greater
    safety. Depth weights are search heuristics, not predicted impurity purge.
    The API target (root object) is never scored as a route impurity.
    """

    scorer_name = "genotox heuristic"

    def __init__(self, config=None, scaler_params=None, late_penalty=2.0):
        super().__init__(config, scaler_params)
        self._reverse_order = True
        self.late_penalty = float(late_penalty)
        if not math.isfinite(self.late_penalty) or self.late_penalty <= 0:
            raise ValueError("late_penalty must be finite and greater than zero")

    def __repr__(self):
        return self.scorer_name

    def _penalty(self, rd_mol, depth):
        """Heuristic for a non-root material, never an exposure/purge estimate."""
        assessment = _assessment(rd_mol)
        if assessment["status"] == "invalid":
            return 0.0
        priority = _heuristic_priority(assessment)
        weight = min(self.late_penalty / (max(depth, 0) + 1), 1.0)
        return max(0.0, 1.0 - priority * weight)

    def _score_node(self, node):
        return self._score_reaction_tree(node.to_reaction_tree())

    def _score_reaction_tree(self, tree):
        score = 1.0
        for mol in tree.molecules():
            if mol is tree.root:
                continue
            depth = tree.depth(mol) // 2
            score *= self._penalty(mol.rd_mol, depth)
        return score

"""Conservative, local RDKit screening for research, not regulatory decisions.

The sparse rules below are custom alerts, not validated mutagenicity models.
The known-material intakes are transcribed from the ICH M7(R2) Addendum's own
summary table, so a match reports that table's figure for this route. No TTC or
ICH M7 class is inferred from a match or from the absence of a match, and a
structural alert never receives a numerical intake.
"""

import math

from rdkit import Chem


_MAX_SMILES_LENGTH = 4096


# FDA's July 2023 adoption of the ICH M7(R2) Addendum, summary table on
# printed pages 12-14 (PDF pages 18-20). A local copy of that table is kept at
# 논문/ICH_M7_R2_Addendum_FDA.pdf. Hydrogen peroxide, dimethyl sulfate and
# p-cresidine appear in the table but are not screened here, because they are
# not in this tool's material list.
ADDENDUM_URL = "https://www.fda.gov/media/170459/download"
# Each row is (name, SMILES, oral or all-other-routes intake in µg/day, route note).
# Values are transcribed from the Addendum's own summary table, which splits
# several compounds by route. This screen covers oral pharmaceutical impurities,
# so the oral or "all other routes" figure is the one stored; where the table
# gives a separate and lower inhalation limit, the note records it so the
# stored figure is never mistaken for a limit that covers every route.
_KNOWN_RAW = (
    ("Acrylonitrile", "C=CC#N", 6.0, None),
    ("Benzyl chloride", "ClCc1ccccc1", 41.0, None),
    ("Bis(chloromethyl)ether", "ClCOCCl", 0.004, None),
    ("1-Chloro-4-nitrobenzene", "O=[N+]([O-])c1ccc(Cl)cc1", 117.0, None),
    ("1,2-Dibromoethane", "BrCCBr", 2.0, None),
    ("Dimethylcarbamyl chloride", "CN(C)C(=O)Cl", 5.0,
     "all other routes; the table gives 0.6 µg/day for inhalation"),
    ("Epichlorohydrin", "ClCC1CO1", 3.0, None),
    ("Ethyl bromide", "CCBr", 32.0, None),
    ("Ethyl chloride", "CCCl", 1810.0, None),
    ("Glycidol", "OCC1CO1", 4.0, None),
    ("Hydrazine", "NN", 39.0,
     "all other routes; the table gives 0.2 µg/day for inhalation"),
    ("Methyl chloride", "CCl", 1361.0, None),
    ("Styrene", "C=Cc1ccccc1", 154.0, None),
    ("Aniline", "Nc1ccccc1", 720.0, "threshold-based PDE, not a TD50 extrapolation"),
    ("Acetaldehyde", "CC=O", 2000.0,
     "oral; the table gives 185 µg/day for all other routes"),
    ("Formaldehyde", "C=O", 10000.0,
     "all other routes; inhalation is limited to 8,000 µg/day or 215 ppb, whichever is lower"),
    ("Vinyl acetate", "C=COC(C)=O", 2000.0,
     "oral; the table gives 758 µg/day for all other routes"),
    ("p-Chloroaniline", "Nc1ccc(Cl)cc1", 34.0,
     "based on liver tumours whose mutagenic mode of action cannot be ruled out"),
)

# Each tuple is (label, alternative SMARTS, special-concern flag). These
# patterns are deliberately not presented as ICH-endorsed rules or QSAR.
_ALERTS_RAW = (
    ("N-nitroso", ("[#7][NX2]=[OX1]", "[#7+]=[NX2][O-]"), True),
    ("Azoxy (special-concern review)", ("[#6]N=[N+]([O-])[#6]",), True),
    ("Aziridine", ("C1NC1",), False),
    ("Epoxide", ("C1OC1",), False),
    ("Sulfonate ester", ("[#6]OS(=O)(=O)[#6]",), False),
    ("Aromatic nitro", ("c[N+](=O)[O-]",), False),
    ("Aromatic amine", ("[NX3;H2,H1;!$(NC=O)]c",), False),
    ("Alkyl halide (Cl/Br/I)", ("[CX4;H2,H3][Cl,Br,I]",), False),
    ("Hydrazine", ("[NX3][NX3]",), False),
    ("Azo", ("[#6]N=N[#6]",), False),
    ("Michael acceptor", ("[CX3]=[CX3][CX3](=O)",), False),
    ("Aldehyde", ("[CX3H1](=O)[#6]", "[CX3H2]=O"), False),
)

_LIMITATIONS = (
    "Research screening only: not an ICH M7 classification, safety determination, "
    "regulatory approval, or impurity control strategy.",
    "Custom sparse SMARTS are not a validated mutagenicity model or two "
    "complementary (Q)SAR methodologies; false positives and false negatives are possible.",
    "No alert detected does not mean safe or Class 5; other toxicities are not assessed.",
    "Atom maps are cleared and canonicalization preserves stereochemistry and charge. "
    "Canonicalization alone cannot establish salt, protonation, tautomer, isotope "
    "or stereoisomer equivalence; these forms are not merged or neutralized.",
    "Special-concern coverage is incomplete, including aflatoxin-like substances; "
    "absence of a detected motif cannot exclude special concern.",
    "No residual concentration, actual exposure, purge, yield or cost is predicted.",
)


def _normalize_material(smiles):
    if not isinstance(smiles, str) or len(smiles) > _MAX_SMILES_LENGTH:
        return None, None
    text = smiles.strip()
    if not text or "\x00" in text or any(char.isspace() for char in text):
        return None, None
    params = Chem.SmilesParserParams()
    params.parseName = False
    params.allowCXSMILES = False
    try:
        mol = Chem.MolFromSmiles(text, params)
        if mol is None or mol.GetNumAtoms() == 0 or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return None, None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol, _canonical(mol)
    except (ValueError, RuntimeError):
        return None, None


def _canonical(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_smiles(smiles: str) -> str | None:
    """Return map-free isomeric canonical SMILES, or None for invalid input.

    Charges and disconnected components are preserved. This deliberately does
    not neutralize salts, select a largest fragment or canonicalize tautomers.
    The lookup table and assess_material use this same normalization pipeline.
    More than 4096 raw characters are rejected before trimming or parsing.
    """
    return _normalize_material(smiles)[1]


_KNOWN = {}
for _name, _smiles, _intake, _route_note in _KNOWN_RAW:
    _key = canonicalize_smiles(_smiles)
    if _key is None:
        raise RuntimeError(f"Invalid known-material SMILES: {_name}")
    _KNOWN[_key] = (_name, _intake, _route_note)

_ALERTS = []
for _label, _smarts_options, _special in _ALERTS_RAW:
    _patterns = tuple(Chem.MolFromSmarts(smarts) for smarts in _smarts_options)
    if any(pattern is None for pattern in _patterns):
        raise RuntimeError(f"Invalid alert SMARTS: {_label}")
    _ALERTS.append((_label, _patterns, _special))

def assess_material(smiles: str) -> dict:
    """Screen a complete SMILES, preserving separate disconnected components.

    A disconnected material can identify known components, but never inherits a
    single component's numerical intake. Charged fragments are not matched to
    neutral parent entries by assuming salt or protonation equivalence.
    Overlength raw input remains invalid and unassessed, without truncation.
    """
    result = {
        "input_smiles": smiles,
        "canonical_smiles": None,
        "status": "invalid",
        "name": None,
        "alerts": [],
        "special_concern": False,
        "intake_ug_day": None,
        "intake_route_note": None,
        "intake_basis": "Invalid or empty structure; no assessment or intake established.",
        "source_url": None,
        "limitations": list(_LIMITATIONS),
    }
    mol, canonical_smiles = _normalize_material(smiles)
    if mol is None:
        if isinstance(smiles, str) and len(smiles) > _MAX_SMILES_LENGTH:
            result["limitations"].append(
                "Raw SMILES exceeds 4096 characters; not parsed or assessed. "
                "The original input was not truncated."
            )
        result["limitations"].append("A non-empty, complete, sanitizable SMILES is required.")
        return result

    result["canonical_smiles"] = canonical_smiles
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    known = []
    for fragment in fragments:
        fragment_smiles = _canonical(fragment)
        entry = _KNOWN.get(fragment_smiles)
        if entry is not None:
            known.append(entry)
            if len(fragments) > 1:
                result["limitations"].append(
                    f"Known component: {entry[0]} ({fragment_smiles}); "
                    "this is not an identification of the entire disconnected material."
                )

    for label, patterns, special in _ALERTS:
        if any(mol.HasSubstructMatch(pattern) for pattern in patterns):
            result["alerts"].append(label)
            result["special_concern"] = result["special_concern"] or special

    if known:
        result["status"] = "known_concern"
        result["name"] = "; ".join(sorted({name for name, _, _ in known}))
        result["source_url"] = ADDENDUM_URL
        result["intake_basis"] = (
            "ICH M7(R2) Addendum-listed compound; no vetted intake for this supplied "
            "material in this screening tool. Listing does not assign an ICH M7 class."
        )
    elif result["alerts"]:
        result["status"] = "structural_alert"
        result["intake_basis"] = (
            "Custom structural alert only; no numerical intake or automatic TTC assigned. "
            "Expert assessment and supporting data are required."
        )
    else:
        result["status"] = "no_alert_detected"
        result["intake_basis"] = (
            "No custom alert detected; no intake established. Not evidence of safety or Class 5."
        )

    if len(fragments) > 1:
        result["limitations"].append(
            "Disconnected components were screened individually. Mixture composition, "
            "counterion equivalence and a whole-material intake are not established; "
            "no component intake is transferred to this material."
        )
    if result["special_concern"]:
        result["intake_basis"] = (
            "Detected special-concern motif: compound-specific expert assessment required. "
            "No TTC fallback and no numerical intake assigned."
        )
        result["limitations"].append(
            "N-nitroso and other detected special-concern motifs must not use a generic TTC. "
            "A motif does not establish identity, potency or a compound-specific intake."
        )
    elif len(fragments) == 1 and known and known[0][1] is not None:
        name, intake, route_note = known[0]
        result["intake_ug_day"] = intake
        result["intake_route_note"] = route_note
        basis = (f"ICH M7(R2) Addendum summary table: {name}, {intake:g} µg/day; "
                 "lifetime oral impurity reference only.")
        if route_note:
            basis += f" Table qualifier: {route_note}."
        result["intake_basis"] = basis
        result["source_url"] = ADDENDUM_URL + "#page=18"
        result["limitations"].append(
            "This lifetime oral impurity reference is not an API dose, measured residual "
            "limit or regulatory compliance decision. Identity, route, duration and "
            "applicability require expert validation; no less-than-lifetime adjustment is made."
        )
    return result


def _finite_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def concentration_limit(intake_ug_day, daily_dose_mg):
    """Convert a supplied intake reference to ppm: 1000 * µg/day / mg/day.

    This is unit conversion, not a determination that an intake applies.
    No conversion is possible when the intake is unknown.
    """
    if intake_ug_day is None:
        return None
    intake = _finite_number(intake_ug_day, "intake_ug_day")
    dose = _finite_number(daily_dose_mg, "daily_dose_mg")
    if intake < 0:
        raise ValueError("intake_ug_day must not be negative")
    if dose <= 0:
        raise ValueError("daily_dose_mg must be greater than zero")
    ppm = (intake / dose) * 1000.0
    if not math.isfinite(ppm):
        raise ValueError("concentration cannot be represented as finite ppm")
    return ppm

"""Reproducible exact identities plus a broader key ONLY for split grouping."""
import hashlib, re, unicodedata
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()

def cas_normalize(value):
    s=unicodedata.normalize('NFKC',str(value or '')).strip()
    s=re.sub(r'[‐‑‒–—−]','-',s)
    s=re.sub(r'^CAS\s*(?:RN|No\.?|Number)?\s*[:#]?\s*','',s,flags=re.I)
    s=re.sub(r'\s+','',s)
    if re.fullmatch(r'\d{5,10}',s):s=s[:-3]+'-'+s[-3:-1]+'-'+s[-1]
    if not re.fullmatch(r'\d{2,7}-\d{2}-\d',s):return '', 'missing_or_format'
    d=s.replace('-','')
    return (s,'valid') if sum((i+1)*int(x) for i,x in enumerate(d[-2::-1]))%10==int(d[-1]) else ('','invalid_checksum')

def normalize_structure(raw):
    raw=str(raw or '').strip()
    empty={'raw_smiles':raw,'canonical_smiles':'','inchikey':'','structure_id':'','leakage_group':'','parent_smiles':'','scaffold':'','status':'missing','fragment_count':0}
    if not raw:return empty
    try:
        mol=Chem.MolFromSmiles(raw)
        if mol is None or not mol.GetNumAtoms():return {**empty,'status':'invalid_smiles'}
        if mol.GetNumHeavyAtoms()>300:return {**empty,'status':'over_300_heavy_atoms'}
        if any(a.GetAtomicNum()==0 for a in mol.GetAtoms()):return {**empty,'status':'wildcard_or_polymer'}
        for a in mol.GetAtoms():a.SetAtomMapNum(0)
        canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
        mol=Chem.MolFromSmiles(canonical)
        assert Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)==canonical
        frags=Chem.GetMolFrags(mol,asMols=True)
        n_organic=sum(any(a.GetAtomicNum()==6 for a in f.GetAtoms()) for f in frags)
        status='valid' if n_organic==1 else ('multiple_organic_components' if n_organic>1 else 'inorganic')
        parent=rdMolStandardize.FragmentParent(mol)
        parent=rdMolStandardize.Uncharger().uncharge(parent)
        Chem.RemoveStereochemistry(parent)
        for a in parent.GetAtoms():a.SetIsotope(0)
        taut=rdMolStandardize.TautomerEnumerator();taut.SetMaxTautomers(256);taut.SetMaxTransforms(512)
        parent=taut.Canonicalize(parent)
        psm=Chem.MolToSmiles(parent,canonical=True,isomericSmiles=False)
        if Chem.MolFromSmiles(psm) is None:
            return {**empty,'canonical_smiles':canonical,'status':'unsupported_parent_standardization'}
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=parent,includeChirality=False)
        return {**empty,'canonical_smiles':canonical,'inchikey':Chem.MolToInchiKey(mol),
                'structure_id':'S_'+digest(canonical)[:24],'leakage_group':'G_'+digest(psm)[:24],
                'parent_smiles':psm,'scaffold':scaffold,'status':status,'fragment_count':len(frags)}
    except Exception as e:return {**empty,'status':'structure_error:'+type(e).__name__}

def condition_category(method):
    s=str(method or '').lower()
    if 'preincubat' in s:return 'preincubation'
    if any(x in s for x in ['standard plate','plate incorporation','plate method']):return 'plate_incorporation'
    if 'fluctuation' in s:return 'fluctuation'
    if any(x in s for x in ['suspension','liquid']):return 'liquid_suspension'
    if any(x in s for x in ['gas','vapor','vapour']):return 'gas_vapour'
    if 'spot' in s:return 'spot'
    return 'other_or_missing'

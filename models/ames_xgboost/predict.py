"""Run a selected Ames pipeline on a SMILES CSV. Never fits an estimator."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd,joblib
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import Descriptors,MACCSkeys,rdFingerprintGenerator
from chemistry import normalize_structure
from features import RECIPES,CORE10,FP_SIZE
from model_lib import score_values
ROOT=Path(__file__).resolve().parent
def featurize(smiles,parents,recipe,descriptor_names):
    funcs=dict(Descriptors._descList)
    if any(n not in funcs for n in descriptor_names):raise RuntimeError('RDKit descriptor definitions differ from training environment')
    gens={k:rdFingerprintGenerator.GetMorganGenerator(radius=3 if k=='ecfp6_2048' else 2,fpSize=FP_SIZE[k],includeChirality=k!='ecfp4_achiral') for k in ['ecfp4_1024','ecfp4_2048','ecfp4_4096','ecfp6_2048','ecfp4_achiral']}
    gens['rdkit']=rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
    mols=[Chem.MolFromSmiles(s) for s in smiles];parts=[]
    for item in RECIPES[recipe]:
        if item in {'desc10','desc2d'}:
            names=CORE10 if item=='desc10' else descriptor_names;arr=np.full((len(mols),len(names)),np.nan)
            for i,m in enumerate(mols):
                for j,n in enumerate(names):
                    try:
                        v=float(funcs[n](m));arr[i,j]=v if np.isfinite(v) and abs(v)<1e20 else np.nan
                    except Exception:pass
            parts.append(sparse.csr_matrix(arr));continue
        key='ecfp4_counts' if item=='ecfp4_logcounts' else item;indices=[];vals=[];ptr=[0]
        for i,m in enumerate(mols):
            if key=='ecfp4_counts':d=gens['ecfp4_2048'].GetCountFingerprint(m).GetNonzeroElements();idx=list(d);v=list(d.values())
            else:
                fp=MACCSkeys.GenMACCSKeys(m) if key=='maccs' else (gens['ecfp4_2048'].GetFingerprint(Chem.MolFromSmiles(parents[i])) if key=='parent_ecfp4' else gens[key].GetFingerprint(m))
                idx=list(fp.GetOnBits());v=[1]*len(idx)
            indices.extend(idx);vals.extend(v);ptr.append(len(indices))
        arr=sparse.csr_matrix((np.asarray(vals,dtype=np.float32),np.asarray(indices,dtype=np.int32),np.asarray(ptr,dtype=np.int32)),shape=(len(mols),FP_SIZE[key]))
        if item=='ecfp4_logcounts':arr.data=np.log1p(arr.data)
        parts.append(arr)
    return sparse.hstack(parts,format='csr',dtype=np.float32)
def predict(frame,modeldir,smiles_column):
    modeldir=Path(modeldir);spec=json.loads((modeldir/'model_spec.json').read_text(encoding='utf-8'))
    meta=json.loads((modeldir/'feature_manifest.json').read_text(encoding='utf-8'))
    import rdkit
    if rdkit.__version__!=meta['rdkit_version']:raise RuntimeError(f'RDKit version must match: {meta["rdkit_version"]}')
    structures=[normalize_structure(s) for s in frame[smiles_column].fillna('').astype(str)]
    good=[i for i,r in enumerate(structures) if r['status']=='valid']
    result=pd.DataFrame({'input_smiles':frame[smiles_column],'canonical_smiles':[r.get('canonical_smiles','') for r in structures],'structure_status':[r['status'] for r in structures],'ames_prediction':pd.array([pd.NA]*len(frame),dtype='Int64'),'ames_score':np.nan,'score_type':''})
    if good:
        X=featurize([structures[i]['canonical_smiles'] for i in good],[structures[i]['parent_smiles'] for i in good],spec['recipe'],meta['descriptor_names'])
        model=joblib.load(modeldir/'model.joblib');result.loc[good,'ames_prediction']=model.predict(X).astype(int);result.loc[good,'ames_score']=score_values(model,X)
        result.loc[good,'score_type']='positive_class_probability_uncalibrated' if hasattr(model,'predict_proba') else 'decision_function_not_probability'
    return result
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--smiles-column',default='smiles');p.add_argument('--model-dir',default=str(ROOT if (ROOT/'model.joblib').exists() else ROOT/'selected_model'));a=p.parse_args()
    df=pd.read_csv(a.input);out=predict(df,a.model_dir,a.smiles_column);out.to_csv(a.output,index=False,encoding='utf-8-sig');print('Predicted',int(out.ames_prediction.notna().sum()),'/',len(out))

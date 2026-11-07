"""Predict with a frozen expanded Ames pipeline; fitting is not performed."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd,joblib,rdkit
from chemistry import normalize_structure
from features import RECIPES
from predict import featurize
HERE=Path(__file__).resolve().parent

def predict_frame(frame,model_dir,smiles_column='smiles',threshold=.5):
    model_dir=Path(model_dir);spec=json.loads((model_dir/'model_spec.json').read_text(encoding='utf-8'))
    meta=json.loads((model_dir/'feature_manifest.json').read_text(encoding='utf-8'))
    if rdkit.__version__!=meta['rdkit_version']:raise RuntimeError('RDKit version differs from training')
    if not 0<=threshold<=1:raise ValueError('threshold must be between 0 and 1')
    structures=[normalize_structure(s) for s in frame[smiles_column].fillna('').astype(str)]
    good=[i for i,r in enumerate(structures) if r['status']=='valid']
    out=pd.DataFrame({'input_smiles':frame[smiles_column].to_numpy(),
        'canonical_smiles':[r.get('canonical_smiles','') for r in structures],
        'structure_status':[r['status'] for r in structures],
        'ames_score':np.nan,'ames_prediction':pd.array([pd.NA]*len(frame),dtype='Int64'),
        'threshold':threshold,'score_type':'positive_class_probability_uncalibrated'})
    if not good:return out
    recipe=spec['recipe']
    if spec.get('blocks'):RECIPES['CUSTOM']=spec['blocks'];recipe='CUSTOM'
    grouped=recipe in {'DESC_FRAGMENTS','DESC_TOPOLOGY'}
    x=featurize([structures[i]['canonical_smiles'] for i in good],
        [structures[i]['parent_smiles'] for i in good],
        'DESC2D' if grouped else recipe,meta['descriptor_names'])
    model=joblib.load(model_dir/'model.joblib')
    if grouped:
        required=model.named_steps['features'].names
        x=x[:,[meta['descriptor_names'].index(name) for name in required]]
    scores=model.predict_proba(x)[:,1]
    out.loc[good,'ames_score']=scores;out.loc[good,'ames_prediction']=(scores>=threshold).astype(int)
    return out

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--smiles-column',default='smiles');p.add_argument('--model-dir',default=str(HERE))
    p.add_argument('--threshold',type=float,default=.5);a=p.parse_args()
    source=pd.read_csv(a.input);result=predict_frame(source,a.model_dir,a.smiles_column,a.threshold)
    result.to_csv(a.output,index=False,encoding='utf-8-sig')
    print(json.dumps({'inputs':len(result),'predicted':int(result.ames_score.notna().sum()),'fit_performed':False}))

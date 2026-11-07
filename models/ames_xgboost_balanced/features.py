"""RDKit features for development molecules only. Never open locked_test."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import Descriptors,MACCSkeys,rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parent
CORE10=['MolWt','MolLogP','TPSA','NumHAcceptors','NumHDonors','NumRotatableBonds','RingCount','NumAromaticRings','FractionCSP3','HeavyAtomCount']
FP_KEYS=['ecfp4_1024','ecfp4_2048','ecfp4_4096','ecfp6_2048','ecfp4_achiral','ecfp4_counts','maccs','rdkit','parent_ecfp4']
FP_SIZE=dict(zip(FP_KEYS,[1024,2048,4096,2048,2048,2048,167,2048,2048]))
RECIPES={
 'DESC10':['desc10'],'DESC2D':['desc2d'],
 'ECFP4':['ecfp4_2048'],'ECFP6':['ecfp6_2048'],'ECFP4_1024':['ecfp4_1024'],'ECFP4_4096':['ecfp4_4096'],
 'ECFP4_COUNT':['ecfp4_counts'],'ECFP4_LOGCOUNT':['ecfp4_logcounts'],'ECFP4_ACHIRAL':['ecfp4_achiral'],
 'MACCS':['maccs'],'RDKIT':['rdkit'],'PARENT_ECFP4':['parent_ecfp4'],
 'ECFP4_DESC10':['ecfp4_2048','desc10'],'ECFP4_DESC2D':['ecfp4_2048','desc2d'],
 'MACCS_DESC2D':['maccs','desc2d'],'RDKIT_DESC2D':['rdkit','desc2d'],
 'ECFP4_ECFP6_DESC2D':['ecfp4_2048','ecfp6_2048','desc2d']
}

def compute_one(smiles,parent_smiles=None):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid normalized SMILES')
    values=[]
    for name,func in Descriptors._descList:
        try:
            value=float(func(mol))
            values.append(value if np.isfinite(value) and abs(value)<1e20 else np.nan)
        except Exception:values.append(np.nan)
    return mol,values

def build(data_dir):
    data_dir=Path(data_dir);out=data_dir/'features';out.mkdir(exist_ok=True)
    df=pd.read_parquet(data_dir/'development_molecules.parquet')
    assert set(df.split)<= {'train','validation'}
    ids=df.structure_id.to_numpy(dtype=str)
    if (out/'manifest.json').exists() and np.array_equal(np.load(out/'structure_ids.npy'),ids):
        print('FEATURES_ALREADY_READY',len(df),flush=True);return
    gens={
      'ecfp4_1024':rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=True),
      'ecfp4_2048':rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True),
      'ecfp4_4096':rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=4096,includeChirality=True),
      'ecfp6_2048':rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048,includeChirality=True),
      'ecfp4_achiral':rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False),
      'rdkit':rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048),
    }
    buf={k:{'index':[],'value':[],'ptr':[0]} for k in FP_KEYS}
    desc=np.full((len(df),len(Descriptors._descList)),np.nan,dtype=np.float64)
    oldindex={};oldparents={};oldfp={};old_desc=None;reused=0;reuse_from=None
    # Fingerprints/descriptors are deterministic, unfitted calculations. Reuse
    # matching development structures only, with identical RDKit definitions.
    for prior in sorted(ROOT.glob('data_v*'),reverse=True):
        folder=prior/'features'
        if prior==data_dir or not (folder/'manifest.json').exists():continue
        meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if meta['rdkit_version']!=__import__('rdkit').__version__ or meta['descriptor_names']!=[n for n,_ in Descriptors._descList]:continue
        oldids=np.load(folder/'structure_ids.npy');oldindex={s:i for i,s in enumerate(oldids)}
        oldframe=pd.read_parquet(prior/'development_molecules.parquet');oldparents=dict(zip(oldframe.structure_id,oldframe.parent_smiles))
        old_desc=np.load(folder/'descriptors.npy',mmap_mode='r');oldfp={k:sparse.load_npz(folder/f'{k}.npz') for k in FP_KEYS};reuse_from=prior.name;break
    if reuse_from:print('FEATURE_CACHE_SOURCE',reuse_from,flush=True)
    for i,row in enumerate(df.itertuples(index=False)):
        if row.structure_id in oldindex and oldparents.get(row.structure_id)==row.parent_smiles:
            j=oldindex[row.structure_id];desc[i]=old_desc[j]
            for key in FP_KEYS:
                a=oldfp[key].getrow(j);b=buf[key];b['index'].extend(a.indices.tolist());b['value'].extend(a.data.tolist());b['ptr'].append(len(b['index']))
            reused+=1
            if (i+1)%2000==0:print(datetime.now().isoformat(timespec='seconds'),f'features {i+1:,}/{len(df):,}; reused {reused:,}',flush=True)
            continue
        mol,vals=compute_one(row.canonical_smiles);desc[i]=vals
        for key in FP_KEYS:
            if key=='ecfp4_counts':
                vec=gens['ecfp4_2048'].GetCountFingerprint(mol).GetNonzeroElements();idx=list(vec);val=list(vec.values())
            else:
                if key=='maccs':fp=MACCSkeys.GenMACCSKeys(mol)
                elif key=='parent_ecfp4':fp=gens['ecfp4_2048'].GetFingerprint(Chem.MolFromSmiles(row.parent_smiles))
                else:fp=gens[key].GetFingerprint(mol)
                idx=list(fp.GetOnBits());val=[1]*len(idx)
            b=buf[key];b['index'].extend(idx);b['value'].extend(val);b['ptr'].append(len(b['index']))
        if (i+1)%1000==0:print(datetime.now().isoformat(timespec='seconds'),f'features {i+1:,}/{len(df):,}',flush=True)
    names=[n for n,_ in Descriptors._descList]
    np.save(out/'structure_ids.npy',ids,allow_pickle=False)
    np.save(out/'descriptors.npy',desc,allow_pickle=False)
    for key,b in buf.items():
        x=sparse.csr_matrix((np.asarray(b['value'],dtype=np.float32),np.asarray(b['index'],dtype=np.int32),np.asarray(b['ptr'],dtype=np.int32)),shape=(len(df),FP_SIZE[key]))
        sparse.save_npz(out/f'{key}.npz',x)
    with (out/'manifest.json').open('w',encoding='utf-8') as f:json.dump({'rows':len(df),'descriptor_names':names,'core10':CORE10,'fp_sizes':FP_SIZE,'recipes':RECIPES,'test_rows':0,'rdkit_version':__import__('rdkit').__version__,'deterministic_cache_rows':reused,'cache_source':reuse_from},f,ensure_ascii=False,indent=2)
    pd.DataFrame(desc[:,[names.index(n) for n in CORE10]],columns=CORE10).assign(structure_id=ids).to_csv(out/'physchem10.csv',index=False,encoding='utf-8-sig')
    print('FEATURES_DONE',len(df),flush=True)

def load_recipe(data_dir,recipe):
    folder=Path(data_dir)/'features'
    meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    ds=None;parts=[];names=[]
    for item in RECIPES[recipe]:
        if item in {'desc10','desc2d'}:
            if ds is None:ds=np.load(folder/'descriptors.npy')
            indices=[meta['descriptor_names'].index(n) for n in CORE10] if item=='desc10' else list(range(ds.shape[1]))
            parts.append(sparse.csr_matrix(ds[:,indices]));names.extend(meta['descriptor_names'][i] for i in indices)
        else:
            key='ecfp4_counts' if item=='ecfp4_logcounts' else item
            x=sparse.load_npz(folder/f'{key}.npz')
            if item=='ecfp4_logcounts':x.data=np.log1p(x.data)
            parts.append(x);names.extend(f'{item}_{i}' for i in range(x.shape[1]))
    x=sparse.hstack(parts,format='csr',dtype=np.float32)
    return x,names

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--data-dir');a=p.parse_args()
    d=a.data_dir or json.loads((ROOT/'active_dataset.json').read_text(encoding='utf-8'))['directory']
    build(d)

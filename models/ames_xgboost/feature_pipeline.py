"""Training-fold-only selection and numeric encoding for molecular descriptors."""
import warnings
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, QuantileTransformer, OneHotEncoder, KBinsDiscretizer
from sklearn.ensemble import ExtraTreesClassifier

FP_PREFIXES=('ecfp','maccs_','rdkit_','parent_ecfp')
def array(x): return x.toarray() if sparse.issparse(x) else np.asarray(x)

class FeatureProcessor(BaseEstimator,TransformerMixin):
    def __init__(self,names,method='anova',k='all',descriptor_k='all',fingerprint_k='all',
                 encoding='float',seed=20261105,threads=4):
        self.names=names;self.method=method;self.k=k;self.descriptor_k=descriptor_k
        self.fingerprint_k=fingerprint_k;self.encoding=encoding;self.seed=seed;self.threads=threads
    def fit(self,X,y):
        self.imputer_=SimpleImputer(strategy='median',keep_empty_features=True)
        a=self.imputer_.fit_transform(X)
        self.variance_=VarianceThreshold().fit(a)
        b=self.variance_.transform(a)
        base=np.flatnonzero(self.variance_.get_support())
        names=np.asarray(self.names,dtype=str)[base]
        fp=np.asarray([n.startswith(FP_PREFIXES) for n in names])
        if self.method=='anova':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');importance=f_classif(b.astype(np.float64),y)[0]
        elif self.method=='embedded':
            selector=ExtraTreesClassifier(n_estimators=128,max_depth=16,min_samples_leaf=2,
                max_features='sqrt',random_state=self.seed,n_jobs=self.threads).fit(b,y)
            importance=selector.feature_importances_
        else:raise ValueError(self.method)
        importance=np.nan_to_num(importance,nan=-np.inf,posinf=np.finfo(float).max)
        def choose(indices,number):
            if number=='all': return indices
            return indices[np.argsort(-importance[indices],kind='stable')[:min(int(number),len(indices))]]
        if self.k!='all': keep=choose(np.arange(len(base)),self.k)
        else: keep=np.r_[choose(np.flatnonzero(~fp),self.descriptor_k),choose(np.flatnonzero(fp),self.fingerprint_k)]
        self.indices_=base[np.sort(keep)]
        self.selected_names_=np.asarray(self.names,dtype=str)[self.indices_]
        self.selected_importances_=importance[np.sort(keep)]
        self.fp_positions_=np.flatnonzero([n.startswith(FP_PREFIXES) for n in self.selected_names_])
        self.desc_positions_=np.flatnonzero([not n.startswith(FP_PREFIXES) for n in self.selected_names_])
        selected=a[:,self.indices_];desc=array(selected[:,self.desc_positions_])
        self.encoder_=None;self.categorical_positions_=np.array([],dtype=int)
        self.numeric_positions_=np.arange(desc.shape[1]);dn=self.selected_names_[self.desc_positions_]
        encnames=dn.tolist()
        if self.encoding in {'bins5_onehot','bins10_onehot'} and desc.shape[1]:
            self.encoder_=KBinsDiscretizer(n_bins=5 if self.encoding=='bins5_onehot' else 10,
                encode='onehot',strategy='quantile',quantile_method='averaged_inverted_cdf',subsample=None)
            self.encoder_.fit(desc);encnames=self.encoder_.get_feature_names_out(dn).tolist()
        elif self.encoding=='counts_onehot' and desc.shape[1]:
            cats=[]
            for i in range(desc.shape[1]):
                unique=np.unique(desc[:,i]);
                if len(unique)<=24 and np.allclose(unique,np.round(unique)):cats.append(i)
            self.categorical_positions_=np.asarray(cats,dtype=int)
            self.numeric_positions_=np.asarray([i for i in range(desc.shape[1]) if i not in cats],dtype=int)
            encnames=dn[self.numeric_positions_].tolist()
            if cats:
                self.encoder_=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
                self.encoder_.fit(desc[:,cats]);encnames+=self.encoder_.get_feature_names_out(dn[cats]).tolist()
        elif self.encoding not in {'float','standard','maxabs','signed_log_standard','quantile_normal','counts_onehot','bins5_onehot','bins10_onehot'}:
            raise ValueError(self.encoding)
        self.output_names_=encnames+self.selected_names_[self.fp_positions_].tolist()
        encoded=self._encode(selected)
        self.scaler_=None
        if self.encoding in {'standard','signed_log_standard'}:self.scaler_=StandardScaler(with_mean=False).fit(encoded)
        elif self.encoding=='maxabs':self.scaler_=MaxAbsScaler().fit(encoded)
        elif self.encoding=='quantile_normal':
            self.scaler_=QuantileTransformer(n_quantiles=min(128,encoded.shape[0]),output_distribution='normal',random_state=self.seed).fit(array(encoded))
        self.n_features_in_=X.shape[1]
        return self
    def _encode(self,selected):
        d=array(selected[:,self.desc_positions_]); parts=[]
        if self.encoding.startswith('bins') and self.encoder_ is not None:
            parts.append(self.encoder_.transform(d))
        elif self.encoding=='counts_onehot':
            if len(self.numeric_positions_):parts.append(sparse.csr_matrix(d[:,self.numeric_positions_]))
            if self.encoder_ is not None:parts.append(self.encoder_.transform(d[:,self.categorical_positions_]))
        elif d.shape[1]:parts.append(sparse.csr_matrix(d))
        if len(self.fp_positions_):parts.append(sparse.csr_matrix(selected[:,self.fp_positions_]))
        x=sparse.hstack(parts,format='csr',dtype=np.float32)
        if self.encoding=='signed_log_standard':x.data=np.sign(x.data)*np.log1p(np.abs(x.data))
        return x
    def transform(self,X):
        a=self.imputer_.transform(X)[:,self.indices_];x=self._encode(a)
        if self.scaler_ is not None:
            x=self.scaler_.transform(array(x) if self.encoding=='quantile_normal' else x)
        return x
    def get_feature_names_out(self,input_features=None):return np.asarray(self.output_names_,dtype=str)

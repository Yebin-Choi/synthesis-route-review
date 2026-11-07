"""Fold-local preprocessing and estimators. No test data API exists here."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_classif
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,QuantileTransformer,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,accuracy_score,confusion_matrix

def stable_f_classif(X,y):
    # Some valid graph descriptors have large magnitudes; accumulate ANOVA in
    # float64 while keeping stored feature matrices compact.
    return f_classif(X.astype(np.float64,copy=False),y)

class SafeSelect(BaseEstimator,TransformerMixin):
    def __init__(self,k='all'):self.k=k
    def fit(self,X,y=None):
        k='all' if self.k=='all' else min(int(self.k),X.shape[1])
        self.selector_=SelectKBest(stable_f_classif,k=k).fit(X,y);return self
    def transform(self,X):return self.selector_.transform(X)
    def get_support(self):return self.selector_.get_support()
    def get_feature_names_out(self,input_features=None):return np.asarray(input_features)[self.get_support()]

class SafeQuantile(BaseEstimator,TransformerMixin):
    def __init__(self,seed=20261105):self.seed=seed
    def fit(self,X,y=None):
        self.qt_=QuantileTransformer(n_quantiles=min(128,X.shape[0]),output_distribution='normal',random_state=self.seed).fit(dense(X));return self
    def transform(self,X):return self.qt_.transform(dense(X))

def dense(X):return X.toarray() if sparse.issparse(X) else X
def signed_log(X):
    X=X.copy()
    if sparse.issparse(X):X.data=np.sign(X.data)*np.log1p(np.abs(X.data))
    else:X=np.sign(X)*np.log1p(np.abs(X))
    return X

def make_pipeline(algorithm,params,spec,y):
    seed=int(spec.get('seed',20261105));threads=int(spec.get('threads',4))
    p=dict(params);balanced=p.pop('balanced',False)
    weight='balanced' if balanced else None
    if algorithm=='logistic':
        model=LogisticRegression(C=float(p.get('C',1)),l1_ratio=0.0,solver='liblinear',max_iter=3000,class_weight=weight,random_state=seed)
    elif algorithm in {'rf','extra_trees'}:
        klass=RandomForestClassifier if algorithm=='rf' else ExtraTreesClassifier
        model=klass(n_estimators=int(p.get('n_estimators',200)),max_depth=p.get('max_depth'),min_samples_leaf=int(p.get('min_samples_leaf',1)),max_features=p.get('max_features','sqrt'),class_weight=weight,n_jobs=threads,random_state=seed)
    elif algorithm=='xgboost':
        from xgboost import XGBClassifier
        ratio=float(np.sum(y==0)/max(np.sum(y==1),1)) if balanced else 1.
        model=XGBClassifier(n_estimators=int(p.get('n_estimators',200)),max_depth=int(p.get('max_depth',4)),learning_rate=float(p.get('learning_rate',.05)),subsample=float(p.get('subsample',.85)),colsample_bytree=float(p.get('colsample_bytree',.8)),reg_lambda=float(p.get('reg_lambda',1)),min_child_weight=float(p.get('min_child_weight',1)),scale_pos_weight=ratio,objective='binary:logistic',eval_metric='auc',tree_method='hist',n_jobs=threads,random_state=seed,verbosity=0)
    elif algorithm=='lightgbm':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=int(p.get('n_estimators',200)),num_leaves=int(p.get('num_leaves',31)),max_depth=-1,learning_rate=float(p.get('learning_rate',.05)),min_child_samples=int(p.get('min_child_samples',20)),colsample_bytree=float(p.get('colsample_bytree',.8)),reg_lambda=float(p.get('reg_lambda',1)),class_weight=weight,n_jobs=threads,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)
    elif algorithm=='svm':
        model=SVC(C=float(p.get('C',1)),gamma=p.get('gamma','scale'),kernel='rbf',class_weight=weight,cache_size=384,max_iter=6000,probability=False,random_state=seed)
    elif algorithm=='hist_gb':
        model=HistGradientBoostingClassifier(max_iter=int(p.get('n_estimators',200)),learning_rate=float(p.get('learning_rate',.05)),max_leaf_nodes=int(p.get('num_leaves',31)),l2_regularization=float(p.get('reg_lambda',1)),class_weight=weight,early_stopping=False,random_state=seed)
    else:raise ValueError(algorithm)
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('variance',VarianceThreshold()),('select',SafeSelect(spec.get('select_k','all')))]
    scaling=spec.get('scaling','standard' if algorithm in {'logistic','svm'} else 'none')
    if scaling=='standard':steps.append(('scale',StandardScaler(with_mean=False)))
    elif scaling=='maxabs':steps.append(('scale',MaxAbsScaler()))
    elif scaling=='quantile':steps.append(('scale',SafeQuantile(seed)))
    elif scaling=='signed_log_standard':steps.extend([('log',FunctionTransformer(signed_log,accept_sparse=True)),('scale',StandardScaler(with_mean=False))])
    elif scaling!='none':raise ValueError(scaling)
    if algorithm in {'xgboost','hist_gb'}:steps.append(('dense',FunctionTransformer(dense,accept_sparse=True)))
    steps.append(('model',model))
    return Pipeline(steps)

def search_space(algorithm,tune):
    if algorithm=='logistic':return {'C':tune.loguniform(.01,100),'balanced':tune.choice([False,True])},{'C':1.,'balanced':False}
    if algorithm in {'rf','extra_trees'}:
        return {'n_estimators':tune.choice([100,200,400,600]),'max_depth':tune.choice([12,24,40,None]),'min_samples_leaf':tune.choice([1,2,4,8]),'max_features':tune.choice(['sqrt',.2,.5]),'balanced':tune.choice([False,True])},{'n_estimators':100,'max_depth':24,'min_samples_leaf':1,'max_features':'sqrt','balanced':False}
    if algorithm=='xgboost':
        return {'n_estimators':tune.choice([100,200,400,600]),'max_depth':tune.randint(3,9),'learning_rate':tune.loguniform(.02,.2),'subsample':tune.uniform(.7,1.),'colsample_bytree':tune.uniform(.5,1.),'reg_lambda':tune.loguniform(.1,20),'min_child_weight':tune.loguniform(1,12),'balanced':tune.choice([False,True])},{'n_estimators':100,'max_depth':4,'learning_rate':.08,'subsample':.85,'colsample_bytree':.8,'reg_lambda':1.,'min_child_weight':1.,'balanced':False}
    if algorithm=='lightgbm':
        return {'n_estimators':tune.choice([100,200,400,600]),'num_leaves':tune.choice([15,31,63]),'learning_rate':tune.loguniform(.02,.2),'min_child_samples':tune.choice([10,20,40,80]),'colsample_bytree':tune.uniform(.5,1.),'reg_lambda':tune.loguniform(.1,20),'balanced':tune.choice([False,True])},{'n_estimators':100,'num_leaves':31,'learning_rate':.08,'min_child_samples':20,'colsample_bytree':.8,'reg_lambda':1.,'balanced':False}
    if algorithm=='svm':return {'C':tune.loguniform(.05,100),'gamma':tune.choice(['scale',.0001,.001,.01,.1]),'balanced':tune.choice([False,True])},{'C':1.,'gamma':'scale','balanced':False}
    if algorithm=='hist_gb':return {'n_estimators':tune.choice([100,200,400]),'num_leaves':tune.choice([15,31,63]),'learning_rate':tune.loguniform(.02,.2),'reg_lambda':tune.loguniform(.1,20),'balanced':tune.choice([False,True])},{'n_estimators':100,'num_leaves':31,'learning_rate':.08,'reg_lambda':1.,'balanced':False}
    raise ValueError(algorithm)

def score_values(model,X):
    if hasattr(model,'predict_proba'):return model.predict_proba(X)[:,1]
    return model.decision_function(X)

def metrics(y,score,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':float(roc_auc_score(y,score)),'average_precision':float(average_precision_score(y,score)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'accuracy':float(accuracy_score(y,pred)),
            'sensitivity':float(tp/max(tp+fn,1)),'specificity':float(tn/max(tn+fp,1)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
